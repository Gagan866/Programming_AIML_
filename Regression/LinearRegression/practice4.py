import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("DataSets/CarPrice_Assignment.csv",index_col=0)

# print(df[["age","price"]])
print(df.head())
print(df.isna().sum())
print(df.info())
print(df["doornumber"].unique())

lable_encode = LabelEncoder()

# df["fueltype"] = lable_encode.fit_transform(df["fueltype"])
# print(df["fueltype"].unique())
# print(df["fueltype"])


df["doornumber"] = lable_encode.fit_transform(df["doornumber"])
print(df["doornumber"].unique())
print(df["doornumber"])



x = df[["doornumber"]]
y = df["price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=42)

print(x_train)
print("-"*20)
print(x_test)
print("-"*20)
print(y_train)
print("-"*20)
print(y_test)
print("-"*20)


model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


result = pd.DataFrame()
result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred
print(result.head(10))


mse = mean_squared_error(y_test,y_pred)
print(mse)


r2 = r2_score(y_test,y_pred)
print(r2)


plt.scatter(x_test,y_test,color = "blue",label = "Actual")
plt.plot(x_test,y_pred,color="Red",label="Predicted")
# sns.regplot(x=x_test,y=y_pred)
plt.legend()
plt.xlabel("Door")
plt.ylabel("Price")
plt.show()