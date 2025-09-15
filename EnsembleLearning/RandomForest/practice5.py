 # 📦 1. Import libraries
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# 🎯 2. Load dataset (example: Iris dataset from seaborn)
data = sns.load_dataset("iris")

# 🛠️ 3. Basic preprocessing
# Drop useless columns if any (example: ID, duplicates)
# data = data.drop(["id"], axis=1)   # only if ID column exists

# Handle missing values (RF can't handle NaNs)
data = data.dropna()

# Encode categorical target if it's object/string
if data["species"].dtype == "object" or data["species"].dtype.name == "category":
    le = LabelEncoder()
    data["species"] = le.fit_transform(data["species"])

# 🚀 4. Features (X) and Target (y)
X = data.drop("species", axis=1)
y = data["species"]

# 🧪 5. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🌲 6. Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# ⚙️ 7. Hyperparameter grid for ensemble tuning
param_grid = {
    "n_estimators": [100, 200, 500],       # number of trees
    "max_depth": [5, 10, None],            # tree depth
    "max_features": ["sqrt", "log2"],      # features per split
    "min_samples_split": [2, 5, 10],       # min samples to split a node
    "min_samples_leaf": [1, 2, 4]          # min samples at leaf node
}

# 🔍 8. GridSearchCV for best parameters
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# ✅ 9. Get best model
best_rf = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# 📊 10. Evaluate on test data
y_pred = best_rf.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:\n", classification_report(y_test, y_pred))

# 📈 11. Feature Importance (interpretability)
importances = best_rf.feature_importances_
features = X.columns

plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
