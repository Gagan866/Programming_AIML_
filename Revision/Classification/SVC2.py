import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.mstats import winsorize

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

print(df.head())
print(df.describe())
print(df.info())
print(df.isna().sum())

# ---------------------------------------------------
# 2. Outlier Visual Inspection
# ---------------------------------------------------
sns.boxplot(df)
plt.title("Boxplot of Numerical Columns")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------
# 3. Target Encoding
# ---------------------------------------------------
df["Species"] = df["Species"].astype("category")

label = LabelEncoder()
df["Species"] = label.fit_transform(df["Species"])

# ---------------------------------------------------
# 4. Winsorization (remove extreme outliers)
# ---------------------------------------------------
df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# ---------------------------------------------------
# 5. Select Features
# ---------------------------------------------------
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df[x_features], y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# 6. Scaling
# ---------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# 7. GridSearchCV for SVC
# ---------------------------------------------------
param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 5, 10, 20],
    "gamma": ["scale", "auto"],   # only for rbf & poly
    "degree": [2, 3, 4]           # only for poly
}

svc = SVC(probability=True)  # probability=True enables ROC later

grid = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Train GridSearch
grid.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# 8. Best Model & Evaluation
# ---------------------------------------------------
print("\nBest Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

best_model = grid.best_estimator_

# Make Predictions
y_pred = best_model.predict(X_test_scaled)

print("\nTEST Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# 9. Optional: Confusion Matrix Heatmap
# ---------------------------------------------------
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix (SVC)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
