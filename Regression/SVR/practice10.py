import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import SVR


df_train = pd.read_csv("DataSets/L_Reg/train_data.csv")
df_test = pd.read_csv("DataSets/L_Reg/test_data.csv")

# x_train = df_train[["carat"]]
# y_train = df_train["price"]
# x_test = df_test[["carat"]]
# y_test = df_test["price"]

# sv = SVR()
# sv.fit(x_train,y_train)
# y_pred = sv.predict(x_test)


# result = pd.DataFrame()
# result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred
# print(result.head(10))


# mse = mean_squared_error(y_test,y_pred)
# print("MSE : ",mse)

# r2 = r2_score(y_test,y_pred)
# print("R2 : ",r2)
 
 
 
x_train = df_train[['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'cut_label',      
       'color_label', 'clarity_label']]
y_train = df_train["price"]
x_test = df_test[['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'cut_label',      
       'color_label', 'clarity_label']]
y_test = df_test["price"]

print(x_train.columns)
print(x_test.columns)
print("Shape of training dataset : ",x_train.shape)
print("Shape of testing dataset : ",x_test.shape)

sv = SVR()
sv.fit(x_train,y_train)

y_pred = sv.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
print("MSE : ",mse)

r2 = r2_score(y_test,y_pred)
print("R2 : ",r2)

