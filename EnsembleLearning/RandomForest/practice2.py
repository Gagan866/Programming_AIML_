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
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("DataSets/titanic.csv")

print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.columns)
print(df.head())
print(df["Age"].unique())
print(df["Embarked"].unique())
print(df["Cabin"].unique())


df.drop(columns=["Cabin","Name","Fare","Ticket"], inplace=True)  # edits df directly[1][10]


df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df['Embarked'].fillna(df['Embarked'].mode()[0])


encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])


features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
x_features = ['PassengerId', 'Pclass','Sex', 'Age', 'SibSp',
       'Parch', 'Embarked']
y_features = df["Survived"]


x_train,x_test,y_train,y_test = train_test_split(df[x_features],y_features,random_state=42,test_size=.20)


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