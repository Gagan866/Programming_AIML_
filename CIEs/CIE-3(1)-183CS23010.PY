import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

data = pd.read_csv("CIEs/house_data.csv")
print("First 5 Rows : \n",data.head(5))


print("Shape : ",data.shape)
print("Summary Statistics : \n",data.describe())
print("Missing Values count : \n",data.isna().sum())


x_features = ['Size', 'Rooms', 'Age']
y_features = ["Price"]


x_train,x_test,y_train,y_test = train_test_split(data[x_features],data[y_features],random_state=42,test_size=.2)


model = LinearRegression()
model.fit(x_train,y_train)

print("Intercept : ",model.intercept_)
print("Coefficient : ",model.coef_)

y_pred = model.predict(x_test)


print("Mean Absolute Error (MAE) : ",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error (MSE) : ",mean_squared_error(y_test,y_pred))
print("R2 Score (R2) : ",r2_score(y_test,y_pred))









