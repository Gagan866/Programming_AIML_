import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("DataSets/Boston.csv",index_col=0)

print(df[["age","medv"]])

# print(df.isna().sum())


x = df[["age"]]
y = df["medv"]

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


# plt.scatter(x_test,y_test,color = "blue",label = "Actual")
# plt.plot(x_test,y_pred,color="Red",label="Predicted")
sns.regplot(x=x_test,y=y_pred)
plt.legend()
plt.xlabel("age")
plt.ylabel("Cost")
plt.show()