import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV


iris_data = sns.load_dataset("iris")

print(iris_data.head())

print(iris_data.isna().sum())

print(iris_data.info())

iris_data["species"] = iris_data["species"].astype("category")

print(iris_data.info())

x=iris_data.drop("species",axis=1)
y=iris_data["species"]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,stratify=y)

rf = RandomForestClassifier(random_state=42,)

param_grid = {
    "n_estimators":[100,200,500],
    "max_depth" : [5,6,10,None],
    "max_features" : ["sqrt","log2"]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(x_train,y_train)

best = grid.best_estimator_
print(grid.best_params_)
print(grid.best_estimator_)
print(best.feature_importances_)

y_pred = best.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))