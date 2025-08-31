import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("DataSets/breast-cancer.csv",index_col = 0)

print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.columns)
print(df.head())


features = [ 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
x_features = [ 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
y_features = ["diagnosis"]


def winsorize_data(data,limits=(0.02,0.02)):
    for col in data.select_dtypes(include=["float64"]):
        data.loc[:,col] = winsorize(data[col], limits=limits)
    return data

df_win = winsorize_data(df)
X = df_win[x_features]
y = df_win["diagnosis"]

# Define features and target
X = df_win[x_features]   # features
y = df_win["diagnosis"]   # target (Series, not list)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


x_train,x_test,y_train,y_test = train_test_split(df[x_features],df[y_features],random_state=42,stratify=df["diagnosis"],test_size=.20)


model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuraccy Report : \n ",accuracy_score(y_test,y_pred))
print("Confusion Matrix : \n ",confusion_matrix(y_test,y_pred))
print("Classification Report : \n ", classification_report(y_test,y_pred))




# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],          # number of trees
    "max_depth": [None, 4 ,5 ,6],          # tree depth
    "max_features": ["sqrt", "log2", None],   # features considered at each split
    "bootstrap": [True, False]                # whether to bootstrap
}

# Model
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                # 5-fold cross-validation
    n_jobs=-1,           # use all processors
    verbose=2,           # show progress
    scoring="accuracy"   # metric to optimize
)

# Fit model
grid_search.fit(x_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_rf = grid_search.best_estimator_
print("Test Accuracy:", best_rf.score(x_test,y_test))