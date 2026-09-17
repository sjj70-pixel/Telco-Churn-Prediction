import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TARGET_RECALL = 0.75


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChurnNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # BCEWithLogitsLoss expects raw logits, so there is no sigmoid here.
        return self.network(x).squeeze(1)


def make_loader(X, y, batch_size, shuffle, pin_memory):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def predict_probabilities(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y_batch.numpy())

    return np.concatenate(probabilities), np.concatenate(targets)


def main():
    set_seeds(RANDOM_STATE)

    base_path = Path(__file__).parent.parent
    file_path = base_path / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(file_path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(["Churn", "customerID", "TotalCharges"], axis=1)
    y = df["Churn"]

    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns
    numerical_cols = X.select_dtypes(include=["number"]).columns

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    # Fit preprocessing only on the training partition to prevent leakage.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    y_train_array = y_train.to_numpy(dtype=np.float32)
    y_val_array = y_val.to_numpy(dtype=np.float32)
    y_test_array = y_test.to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Number of processed features: {X_train_processed.shape[1]}")

    train_loader = make_loader(
        X_train_processed,
        y_train_array,
        BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        X_val_processed,
        y_val_array,
        BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        X_test_processed,
        y_test_array,
        BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = ChurnNN(input_size=X_train_processed.shape[1]).to(device)

    # The distribution of churners vs nonchurners is large, so need to weigh the cross-entropy
    negatives = float((y_train_array == 0).sum())
    positives = float((y_train_array == 1).sum())
    pos_weight = torch.tensor([negatives / positives], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(X_batch)

        train_loss = running_loss / len(train_loader.dataset)
        y_val_prob, y_val_observed = predict_probabilities(model, val_loader, device)
        val_auc = roc_auc_score(y_val_observed, y_val_prob)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss: {train_loss:.4f} | "
            f"validation ROC-AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation ROC-AUC: {best_val_auc:.4f}")

    # Pick the largest ROC threshold that achieves the target validation recall
    y_val_prob, y_val_observed = predict_probabilities(model, val_loader, device)
    _, tpr, thresholds = roc_curve(y_val_observed, y_val_prob)
    qualifying_indices = np.flatnonzero(tpr >= TARGET_RECALL)
    threshold_index = qualifying_indices[0]
    threshold = thresholds[threshold_index]
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Validation recall: {tpr[threshold_index]:.4f}")

    # Use the model on the test set
    y_test_prob, y_test_observed = predict_probabilities(model, test_loader, device)
    y_test_pred = (y_test_prob >= threshold).astype(int)

    print(f"Test accuracy: {accuracy_score(y_test_observed, y_test_pred):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test_observed, y_test_prob):.4f}")
    print(classification_report(y_test_observed, y_test_pred, digits=4))

    RocCurveDisplay.from_predictions(y_test_observed, y_test_prob)
    plt.title("PyTorch Neural Network ROC Curve")
    plt.show()


if __name__ == "__main__":
    main()
