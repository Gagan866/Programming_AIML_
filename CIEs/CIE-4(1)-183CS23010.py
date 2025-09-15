import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

df = pd.read_csv("CIEs/loan_default_dataset.csv")

print("_"*40)
print(" "*10,"EDA")
print("_"*40)
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.columns)
print("_"*40)
print("_"*40)

sns.boxplot(df)
plt.show()

print("_"*40)
print("Type Conversion")
print("_"*40)

df["Education"] = df["Education"].astype("category")
df["Property_Area"] = df["Property_Area"].astype("category")
print("_"*40)
print(df.info())
print("_"*40)


education = {"Graduate":1,"Not Graduate":0}
property_area = {"Rural":0,"Urban":1,"Semiurban":2}

print("_"*40)
print("Encoding")
print("_"*40)

df["Education"] = df["Education"].map(education)
df["Property_Area"] = df["Property_Area"].map(property_area)
print("_"*40)
print(df.head(10))
print("_"*40)

x = df.drop("Loan_Status",axis=1)
y = df["Loan_Status"]
# print(x.columns)

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,random_state=42,test_size=0.2)

print("_"*40)
print("Decision Tree Building...")
print("_"*40)

model_dt = DecisionTreeClassifier(criterion="entropy",max_depth=6,random_state=42)

model_dt.fit(x_train,y_train)

y_pred_dt = model_dt.predict(x_test)

print("_"*40)
print("Accuracy : ",accuracy_score(y_test,y_pred_dt))
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred_dt))
print("Classification Report : ",classification_report(y_test,y_pred_dt))
print("_"*40)

print("_"*40)
print("Random Forest Building...")
print("_"*40)

model_rf = RandomForestClassifier(random_state=42,n_estimators=100,max_depth=6,max_features="sqrt")
model_rf.fit(x_train,y_train)

y_pred_rf = model_rf.predict(x_test)

print("_"*40)
print("Accuraccy Report :  ",accuracy_score(y_test,y_pred_rf))
print("Confusion Matrix :  ",confusion_matrix(y_test,y_pred_rf))
print("Classification Report :  ", classification_report(y_test,y_pred_rf))
print("_"*40)


param_grid = {
    "n_estimators": [100, 200,300,400],          
    "max_depth": [4,5,6,7,8,None],          
    "max_features": ["sqrt", "log2"]     
}

print("_"*40)
print("Random Forest With Hyperparameters Building...")
print("_"*40)

model_rfh = RandomForestClassifier(random_state=42)


grid_search = GridSearchCV(
    estimator=model_rfh,
    param_grid=param_grid,
    cv=5,               
    n_jobs=-1,          
    verbose=2,           
    scoring="accuracy"   
)

grid_search.fit(x_train, y_train)


print("_"*40)
print("Best Parameters : ", grid_search.best_params_)
print("Best CV Accuracy : ", grid_search.best_score_)
print("Best Estimator : ", grid_search.best_estimator_)
print("_"*40)


best_rf = grid_search.best_estimator_
print("_"*40)
print("Test Accuracy : ", best_rf.score(x_test,y_test))
print("_"*40)

y_pred_rfh = best_rf.predict(x_test)

print("_"*40)
print("Accuraccy Report :  ",accuracy_score(y_test,y_pred_rfh))
print("Confusion Matrix :  ",confusion_matrix(y_test,y_pred_rfh))
print("Classification Report :  ", classification_report(y_test,y_pred_rfh))
print("_"*40)



print("_"*40)
print("Comparision")
print("_"*40)
print("Decision Tree")
print("_"*40)
print("Accuracy : ",accuracy_score(y_test,y_pred_dt))
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred_dt))
print("Classification Report : ",classification_report(y_test,y_pred_dt))
print("_"*40)
print("RandomForest")
print("_"*40)
print("Accuraccy Report :  ",accuracy_score(y_test,y_pred_rf))
print("Confusion Matrix :  ",confusion_matrix(y_test,y_pred_rf))
print("Classification Report :  ", classification_report(y_test,y_pred_rf))
print("_"*40)
print("Random Forest with Hyperparameters")
print("_"*40)
print("Accuraccy Report :  ",accuracy_score(y_test,y_pred_rfh))
print("Confusion Matrix :  ",confusion_matrix(y_test,y_pred_rfh))
print("Classification Report :  ", classification_report(y_test,y_pred_rfh))
print("_"*40)