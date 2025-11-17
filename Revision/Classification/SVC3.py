import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

# Encode target
df["Species"] = df["Species"].astype("category")
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Winsorize SepalWidthCm
df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# Features & Target
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    df[x_features], y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 2. SVC Pipeline + GridSearchCV
# ------------------------------------------------------------

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])

param_grid = {
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__C": [0.1, 1, 5, 10, 20],
    "svc__gamma": ["scale", "auto"],   # only used for rbf and poly
    "svc__degree": [2, 3, 4]           # only used for poly
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

# Train grid search
grid.fit(X_train, y_train)

# ------------------------------------------------------------
# 3. Best Model + Evaluation
# ------------------------------------------------------------

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
