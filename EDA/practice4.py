
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns



df = pd.read_csv("DataSets/data_loan.csv")

print(df.head())
print(df.info())
num = df.select_dtypes(include="int64")
corr = num.corr()
print(corr)

sns.heatmap(data=corr,annot=True,cmap="crest")

plt.show()
df = pd.read_csv("DataSets/Toyota.csv",index_col=0)

print(df.head())
print(df.info())
print(df.dropna(inplace=True))
num = df.select_dtypes(include=["int64","float64"])
corr = num.corr()
print(corr)

sns.heatmap(data=corr,annot=True,cmap="crest")

plt.show()
df = pd.read_csv("DataSets/bank.csv",index_col=0,sep=";")

print(df.head())
print(df.info())
print(df.dropna(inplace=True))
num = df.select_dtypes(include=["int64","float64"])
corr = num.corr()
print(corr)

sns.heatmap(data=corr,cmap="crest",annot=True)

plt.show()
df = pd.read_csv("DataSets/titanic.csv",index_col=0)

print(df.head())
print(df.info())
print(df.dropna(inplace=True))
num = df.select_dtypes(include=["int64","float64"])
corr = num.corr()
print(corr)

sns.heatmap(data=corr,cmap="crest",annot=True)

plt.show()

# df = pd.read_csv("DataSets/Boston.csv",index_col=0)

# print(df.head())
# print(df.info())
# print(df.dropna(inplace=True))
# num = df.select_dtypes(include=["int64","float64"])
# corr = num.corr()
# print(corr)

# sns.heatmap(data=corr,cmap="crest",annot=True)

# plt.show()