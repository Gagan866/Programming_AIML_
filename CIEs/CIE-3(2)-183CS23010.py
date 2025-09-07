import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,confusion_matrix,accuracy_score)

data = pd.read_csv("CIEs/tumor_data.csv")
print("Data : \n",data.head())


print("Shape of data : \n",data.shape)
print("Columns : ",data.columns)
print("All Features : ",data[['Radius', 'Texture', 'Smoothness']])
print("Target Column : ",data["Tumor_Class"])
print("Missing values : \n",data.isna().sum())



x_features = ['Radius', 'Texture', 'Smoothness']
y_features = ["Tumor_Class"]


x_train,x_test,y_train,y_test = train_test_split(data[x_features],data[y_features],random_state=42,test_size=.2)


model = LogisticRegression()
model.fit(x_train,y_train)

print("Intercept : ",model.intercept_)
print("Coefficient : ",model.coef_)
print("Iterations Taken : ",model.n_iter_)

y_pred = model.predict(x_test)


print("Classification Report (CR) : \n",classification_report(y_test,y_pred))
print("Confusion Matrix (CM) : \n",confusion_matrix(y_test,y_pred))
print("Accuracy Score (AS) : ",accuracy_score(y_test,y_pred))