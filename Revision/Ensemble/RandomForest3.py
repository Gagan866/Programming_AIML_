import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# --------------------------
# 1. Load Dataset
# --------------------------
df = pd.read_csv("Revision/Data/logistic_regression1.csv", index_col=0)

# Convert churn to 1/0
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Feature groups
numeric_cols = ["age", "monthly_charges", "total_charges",
                "tenure", "internet_usage", "customer_support_calls"]

categorical_cols = ["gender", "contract_type"]

X = df[numeric_cols + categorical_cols]
y = df["churn"]

# --------------------------
# 2. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --------------------------
# 3. Preprocessing Pipeline
# --------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# --------------------------
# 4. Full Pipeline (Preprocess + Random Forest)
# --------------------------
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(   
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

# --------------------------
# 5. Train Model
# --------------------------
pipeline.fit(X_train, y_train)

# --------------------------
# 6. Predictions
# --------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# --------------------------
# 7. Evaluation
# --------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_prob))

# --------------------------
# 8. ROC Curve
# --------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --------------------------
# 9. Feature Importance (Replaces Coefficients)
# --------------------------

# Get OHE feature names
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical_cols)

all_features = numeric_cols + list(ohe_features)

# Extract feature importances
importances = pipeline.named_steps["model"].feature_importances_

feat_df = pd.DataFrame({"Feature": all_features, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False)

print("\nTop Important Features:\n", feat_df.head(10))

# Plot importance
plt.figure(figsize=(8,5))
plt.barh(feat_df["Feature"][:10], feat_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.show()
