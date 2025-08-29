import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv("DataSets/L_Reg/diamonds.csv")

print(df.head()) 
print(df.info())
print(df.isna().sum())
print(df["cut"].unique())
print(df["color"].unique())
print(df["clarity"].unique())


# missing_val_1 = df[df["cut"].isna()]
# print(missing_val_1)
# missing_val_2 = df[df["color"].isna()]
# print(missing_val_2)
# missing_val_3 = df[df["clarity"].isna()]
# print(missing_val_3)

# sns.boxplot(data=df,x="carat")
# plt.show()

# sns.boxplot(data=df,x="depth")
# plt.show()

# sns.boxplot(data=df,x="table")
# plt.show()

# sns.boxplot(data=df,x="price")
# plt.show()


df_filled = df.fillna(df.mean(numeric_only=True))
print(df_filled.isna().sum())
print(df_filled.info())
 
df_filled["cut"] = df_filled["cut"].astype("category")
df_filled["color"] = df_filled["color"].astype("category")
df_filled["clarity"] = df_filled["clarity"].astype("category")
print(df_filled.info())



cut = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color = {"D":6,"E":5,"F":4,"G":3,"H":2,"I":1,"J":0}
clarity = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}

df_filled["cut"] = df_filled["cut"].map(cut)
df_filled["color"] = df_filled["color"].map(color)
df_filled["clarity"] = df_filled["clarity"].map(clarity)

print(df_filled.head(10))
print(df_filled.info())

df_filled["cut"] = df_filled["cut"].fillna(df_filled["cut"].mode()[0])
df_filled["color"] = df_filled["color"].fillna(df_filled["color"].mode()[0])
df_filled["clarity"] = df_filled["clarity"].fillna(df_filled["clarity"].mode()[0])
print(df_filled.isna().sum())
print(df_filled.head(10))


# df.dropna(inplace=True)
# print(df.info())
# x = df[["carat"]]
# y = df["price"]

x = df_filled[["carat"]]
y = df_filled["price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=25)

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
print(model.coef_)
print(model.intercept_)
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
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()