import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# load data
df1 = pd.read_csv("DataSets/Iris.csv",index_col = 0)
df = df1


# preview data
print(df.head(10))
print(df.shape)


# assess data
print(df.describe())
print(df.info())
print(df["Species"].unique())
print(df["Species"].value_counts())
print(df.isna().sum())
print(df.duplicated().value_counts())


df["Species"] = df["Species"].astype("category")
print(df.info())
print(df["Species"].unique())


# EDA
sns.pairplot(df,hue="Species")
# plt.show()

sns.boxplot(df)
# plt.show()

sns.boxplot(df["SepalWidthCm"])
# plt.show()


# IQR
Q1 = df["SepalWidthCm"].quantile(0.25)
Q3 = df["SepalWidthCm"].quantile(0.75)
IQR = Q3-Q1
l_l = Q1-1.5*IQR
u_l = Q3+1.5*IQR

outliers = df[(df["SepalWidthCm"]>u_l)|(df["SepalWidthCm"]<l_l)]
print("Out",outliers)


df["SepalWidthCm"] = winsorize(df["SepalWidthCm"],limits=(0.1,0.1))

sns.boxplot(df["SepalWidthCm"])
# plt.show()


species = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

df["Species"] = df["Species"].map(species)
print(df.head(10))


features_corr = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
corr = df[features_corr].corr()

sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()


# Model
x = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = df["Species"]

model = LogisticRegression()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(y_pred)

result = pd.DataFrame()
result["y_test"],result["y_pred"] = y_test,y_pred
print(result)







