import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("DataSets/L_Reg/diamonds.csv")

print(df.head()) 
print(df.info())
print(df.isna().sum())
print(df["cut"].unique())
print(df["color"].unique())
print(df["clarity"].unique())


# Visalizing Data
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


# Filling NUm Data with median
df_filled = df.fillna(df.median(numeric_only=True))
print(df_filled.isna().sum())
print(df_filled.info())


# Type conversion obj to cat
df_filled["cut"] = df_filled["cut"].astype("category")
df_filled["color"] = df_filled["color"].astype("category")
df_filled["clarity"] = df_filled["clarity"].astype("category")
print(df_filled.dropna(inplace=True))
print(df_filled.info())
print(df_filled.isna().sum())


# Zscore outlier detection
# df_filled[["z_carat","z_depth","z_table","z_price","z_x","z_y","z_z"]] = zscore(df_filled[["carat","depth","table","price","x","y","z"]])
# print(df_filled.head(10))


# IQR
# Q1 = df_filled[["carat","depth","table","price","x","y","z"]].quantile(0.25)
# Q3 = df_filled[["carat","depth","table","price","x","y","z"]].quantile(0.75)
# IQR = Q3-Q1
# l_l = Q1-1.5*IQR
# u_l = Q3+1.5*IQR
# print(Q1)
# print(Q3)
# print(IQR)
# print(l_l)
# print(u_l)
# IQR filling
# df_filled[["carat","depth","table","price","x","y","z"]] = np.where(df_filled[["carat","depth","table","price","x","y","z"]]>u_l,u_l,np.where(df_filled[["carat","depth","table","price","x","y","z"]]<l_l,l_l,df_filled[["carat","depth","table","price","x","y","z"]]))


def winsorize_data(data,limits=(0.1,0.1)):
    for col in data.select_dtypes(include=["float64"]):
         data.loc[:,col] = winsorize(df_filled[col], limits=limits)
    return data


# Winsorization
df_win = winsorize_data(df_filled)
# sns.boxplot(data=df_win,x="carat")
# plt.show()
# sns.boxplot(data=df_win,x="depth")
# plt.show()
# sns.boxplot(data=df_win,x="table")
# plt.show()
# sns.boxplot(data=df_win,x="price")
# plt.show()
# sns.boxplot(data=df_win,x="x")
# plt.show()
# sns.boxplot(data=df_win,x="y")
# plt.show()
# sns.boxplot(data=df_win,x="z")
# plt.show()


cut = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color = {"D":6,"E":5,"F":4,"G":3,"H":2,"I":1,"J":0}
clarity = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}
df_win["cut"] = df_win["cut"].map(cut)
df_win["color"] = df_win["color"].map(color)
df_win["clarity"] = df_win["clarity"].map(clarity)
# print(df_filled.head(10))
# print(df_filled.info())


df_corr = df_win.select_dtypes(include=["float64"])
corr = df_corr.corr()
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()


x_train,x_test = train_test_split(df_win,test_size=.2,random_state=42)
print(x_train.shape)
print(x_test.shape)


# def standardise_data(data,columns=None):
#     if columns is None:
#         columns = data.select_dtypes(include=["number"]).columns

#     scaler = StandardScaler()
#     data.loc[:,columns] = scaler.fit_transform(data[columns])
#     return df


scaler = StandardScaler()
# print(x_train.head(10))
x_train[['carat', 'depth', 'table', 'x', 'y', 'z', "price"]] = scaler.fit_transform(x_train[['carat', 'depth', 'table', 'x', 'y', 'z', "price"]])
# print(x_train.head(10))
x_test[['carat', 'depth', 'table', 'x', 'y', 'z', "price"]] = scaler.fit_transform(x_test[['carat', 'depth', 'table', 'x', 'y', 'z', "price"]])

x = x_train[['carat', 'depth', 'table', 'x', 'y', 'z', 'cut','color', 'clarity']]
y = x_train["price"]
x1 = x_test[['carat', 'depth', 'table', 'x', 'y', 'z', 'cut','color', 'clarity']]
y1 = x_test["price"]

model = LinearRegression()
model.fit(x,y)

y_pred = model.predict(x1)

# result = pd.DataFrame()
# result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred
# print(result.head(10))

mse = mean_squared_error(y1,y_pred)
print(mse)
r2 = r2_score(y1,y_pred)
print(r2)