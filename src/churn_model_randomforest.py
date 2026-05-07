import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
)

base_path = Path(__file__).parent.parent
file_path = base_path / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Convert TotalCharges from string to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Find the rows with NaN
# print(df[df["TotalCharges"].isnull()].index)

# Delete rows with NaN (These are rows that haven't had a charge yet)
df = df.dropna()

# Change categories for random forest, if churn is yes, change to 1, if churn is no, change to 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Split features and target, X will be table of features, Y the target
# As totalcharges = monthlycharges x tenure, we have multicollinearity, so we'll drop one, total charges
X = df.drop(["Churn", "customerID", "TotalCharges"], axis=1)
y = df["Churn"]

# Separate string columns and numerical columns
categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns

# Preprocess the features pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough",
)

# Model pipeline
model = Pipeline(
    [
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

# Cross validate the model
# As there are much more non-churners than churners in the dataset, stratify
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print("CV ROC-AUC:", scores)
print("Mean ROC-AUC:", scores.mean())

# Split the data into training data and testing data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split the training data into training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# Train model
model.fit(X_train, y_train)

# Use the validation set to pick threshold
y_val_prob = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
target_recall = 0.75
idx = np.where(tpr >= target_recall)[0][0]
threshold = thresholds[idx]
print("Chosen threshold:", threshold)
print("Validation recall:", tpr[idx])

# Retrain on the full training data set
model.fit(X_train_full, y_train_full)

# Use the found threshold on the test set
y_test_prob = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= threshold).astype(int)

# Display roc curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_test_prob))
print(classification_report(y_test, y_test_pred))

# Gets the importance score for each feature
importances = model.named_steps["classifier"].feature_importances_

# Grab the feature names. OneHotEncoder expanded the feature space
feature_names = model.named_steps["preprocessing"].get_feature_names_out()


# Build a table with names and importance score, order by size
importance_df = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

# Clean the names of the features
importance_df["feature"] = importance_df["feature"].str.replace("cat__", "")
importance_df["feature"] = importance_df["feature"].str.replace("remainder__", "")
print(importance_df.head(15))
