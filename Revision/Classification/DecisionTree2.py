import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats.mstats import winsorize

# ------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

print(df.info())
print(df.describe())

sns.boxplot(df)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# ------------------------------------------------------------
# 2. Encode Target
# ------------------------------------------------------------
df["Species"] = df["Species"].astype("category")
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Winsorize
df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# ------------------------------------------------------------
# 3. Feature Selection
# ------------------------------------------------------------
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(
    df[x_features], y, random_state=42, test_size=0.2, stratify=y
)

# ------------------------------------------------------------
# 4. Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

# ------------------------------------------------------------
# 5. ⭐ GridSearchCV for Decision Tree
# ------------------------------------------------------------
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 2, 3, 4, 5, 7, 10],
    "min_samples_split": [2, 3, 4, 5, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5]
}

dtree = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(
    estimator=dtree,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(x_train_scaled, y_train)

# ------------------------------------------------------------
# 6. Best Model + Results
# ------------------------------------------------------------
print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(x_test_scaled)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.show()
