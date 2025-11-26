import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# 4. Full Pipeline (Preprocessing + Model)
# --------------------------

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
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
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --------------------------
# 9. Extract Coefficients
# --------------------------

# 1) Get feature names after OneHotEncoder
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(categorical_cols)

all_features = numeric_cols + list(ohe_features)

# 2) Get model coefficients
coefs = pipeline.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({"Feature": all_features, "Coefficient": coefs})
coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

print("\nTop Coefficients:\n", coef_df.head(10))


































# Full correct non-pipeline logistic regression (scaler AFTER split)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# --------------------------
# 0. Settings
# --------------------------
RSEED = 42

# --------------------------
# 1. Load data
# --------------------------
df = pd.read_csv("Revision/Data/logistic_regression1.csv", index_col=0)

# Map target to numeric
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Feature lists (adjust if your real dataset differs)
numeric_cols = ["age", "monthly_charges", "total_charges",
                "tenure", "internet_usage", "customer_support_calls"]
categorical_cols = ["gender", "contract_type"]

# --------------------------
# 2. Split BEFORE fitting encoders/scalers (no leakage)
# --------------------------
X = df[numeric_cols + categorical_cols]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RSEED
)

# --------------------------
# 3. OneHotEncode categorical columns (fit on train only)
# --------------------------
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")

X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat  = ohe.transform(X_test[categorical_cols])

# Build DataFrames for encoded cats with proper column names and index
cat_cols_encoded = ohe.get_feature_names_out(categorical_cols)
X_train_cat_df = pd.DataFrame(X_train_cat, columns=cat_cols_encoded, index=X_train.index)
X_test_cat_df  = pd.DataFrame(X_test_cat,  columns=cat_cols_encoded, index=X_test.index)

# --------------------------
# 4. Scale numeric columns (fit scaler on train only)
# --------------------------
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[numeric_cols])
X_test_num  = scaler.transform(X_test[numeric_cols])

X_train_num_df = pd.DataFrame(X_train_num, columns=numeric_cols, index=X_train.index)
X_test_num_df  = pd.DataFrame(X_test_num,  columns=numeric_cols, index=X_test.index)

# --------------------------
# 5. Combine numeric + encoded categorical
# --------------------------
X_train_final = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
X_test_final  = pd.concat([X_test_num_df,  X_test_cat_df],  axis=1)

# Optional: sanity check shapes
print("X_train shape:", X_train_final.shape)
print("X_test  shape:", X_test_final.shape)

# --------------------------
# 6. Train Logistic Regression
# --------------------------
model = LogisticRegression(max_iter=1000, random_state=RSEED)
model.fit(X_train_final, y_train)

# --------------------------
# 7. Predict & Evaluate
# --------------------------
y_pred = model.predict(X_test_final)
y_prob = model.predict_proba(X_test_final)[:, 1]  # positive class probabilities

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:", roc_auc)

# --------------------------
# 8. ROC Curve
# --------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# --------------------------
# 9. (Optional) Inspect model coefficients
# --------------------------
feature_names = list(X_train_final.columns)
coefs = model.coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df = coef_df.reindex(coef_df.coef.abs().sort_values(ascending=False).index)  # sort by absolute impact
print("\nTop coefficients:\n", coef_df.head(10))
