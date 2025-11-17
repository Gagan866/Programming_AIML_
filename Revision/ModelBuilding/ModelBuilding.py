import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Revision\Data\Iris.csv",index_col=0)

print(df)
# print(df.info())
# print(df.describe())
# print(df.isna().sum())
# print(df.duplicated().sum())

df["Species"] = df["Species"].astype("category")
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
print(df)

x_features = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
y_features = df["Species"]

x_train,x_test,y_train,y_test = train_test_split(df[x_features],y_features,test_size=.2)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

hyperparms = {
    "C":[0.01,0.1,1,10,100],
    "penalty":["l2"],
    "solver":["lbfgs"],
    "max_iter":[200]
}

lg = LogisticRegression()

model = GridSearchCV(estimator=lg,param_grid=hyperparms,cv=5,scoring="accuracy",n_jobs=-1)

model.fit(x_train,y_train)

print("_"*20)
print(model.cv_results_)
print("_"*20)
print(model.best_index_)
print("_"*20)
print(model.best_estimator_)
print("_"*20)
print(model.best_params_)
print("_"*20)
    
results = model.cv_results_

best_index = model.best_index_

fold_scores = [
    results[f"split{i}_test_score"][best_index]
    for i in range(5)
]

print("CV Fold Accuracies:", fold_scores)
print("Mean Accuracy:", results["mean_test_score"][best_index])
print("Std Dev:", results["std_test_score"][best_index])


y_pred = model.predict(x_test)

acc = accuracy_score(y_test,y_pred)
clr = classification_report(y_test,y_pred)

print(acc)
print(clr)
# print(acc.mean())
# print(acc.std())

