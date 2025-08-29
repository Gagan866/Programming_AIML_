import time 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score)



df = pd.read_csv("DataSets/titanic.csv")

print(df.head())
print(df.columns)
print(df.info())
print(df.describe())

df.drop(columns=["Name","Cabin","Fare","Ticket"],inplace=True)

df[["Sex","Embarked"]] = df[["Sex","Embarked"]].astype("category")

print(df.info())


df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.info())

encode = LabelEncoder()

df["Embarked"] = encode.fit_transform(df["Embarked"])
df["Sex"] = encode.fit_transform(df["Sex"])

print(df.head())


x = df.drop("Survived",axis=1)
y = df["Survived"]

print(x.shape)


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,stratify=y,test_size=0.2)



models = {
    "Logistic" : LogisticRegression(max_iter=500),
    "SVC": SVC(random_state=42, max_iter=1000),

    "Decision_Tree" : DecisionTreeClassifier(random_state= 42)
}

param_grid = {
    "Logistic" : {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    
},
    "SVC" : {
    'C': [0.1, 1, 10, 100],       # Regularization strength
    'gamma': [1, 0.1, 0.01, 0.001], # Kernel coefficient for RBF
    'kernel': ['rbf', 'poly']
    # Types of kernels
},
    "Decision_Tree" :  {
    "criterion": ["gini", "entropy", "log_loss"],  # impurity measures
    "max_depth": [3, 5, 8, None] # depth of tree
    # "min_samples_split": [2, 5, 10],
    # "min_samples_leaf": [1, 2, 4]
}
}


# grid = GridSearchCV(models["Logistic"],param_grid["Logistic"],cv=5,scoring="accuracy",n_jobs=-1)
# grid.fit(x_train,y_train)
# y_pred = grid.predict(x_test)
# acc = accuracy_score(y_test,y_pred)


# print("Best Params : ", grid.best_params_)
# print("Test CV accuraccy : ",grid.best_score_)
# print("Test accuraccy : ",acc)
# print("Confusion Matrix : \n ",confusion_matrix(y_test,y_pred))
# print("Classification Report : \n ", classification_report(y_test,y_pred))




result = {}

for name,model in models.items():
    print(f"\n Training {name}...")
    start = time.time()
    
    grid = GridSearchCV(model,param_grid=param_grid[name],cv=5,scoring="accuracy",n_jobs=-1)
    grid.fit(x_train,y_train)
    
    y_pred = grid.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    
    end = time.time()
    
    result[name] = {
        "Best Params" : grid.best_params_,
        "Train CV Accuracy" : grid.best_score_,
        "Train Accuracy" : acc,
        "Time Taken" : round(end - start , 2)
    }
    
    print(f"{name} Results : ")
    print("Best Params : ", grid.best_params_)
    print("Test CV accuraccy : ",grid.best_score_)
    print("Test accuraccy : ",acc)
    print("Confusion Matrix : \n ",confusion_matrix(y_test,y_pred))
    print("Classification Report : \n ", classification_report(y_test,y_pred))
    
    
print("\n Final Comparision : ")
for model,res in result.items():
    print(f"{model} : {res}")