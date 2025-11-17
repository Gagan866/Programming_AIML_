import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize

# ------------------------------------------------------------
# 1. Load & Prepare Dataset
# ------------------------------------------------------------
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

df["Species"] = df["Species"].astype("category")
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# Feature selection
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    df[x_features], y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# ⭐ Pipeline + GridSearchCV (Random Forest Version)
# ------------------------------------------------------------

pipe = Pipeline([
    ("scaler", StandardScaler()),       # scaling not required, but matching your structure
    ("rf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "rf__n_estimators": [50, 100, 150, 200],
    "rf__criterion": ["gini", "entropy"],
    "rf__max_depth": [None, 2, 3, 4, 5, 7, 10],
    "rf__min_samples_split": [2, 3, 4, 5, 10],
    "rf__min_samples_leaf": [1, 2, 3, 4, 5],
    "rf__bootstrap": [True, False]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# ------------------------------------------------------------
# Predict Using Best Estimator
# ------------------------------------------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
