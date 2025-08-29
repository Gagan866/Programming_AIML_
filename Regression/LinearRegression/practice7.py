import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv("DataSets/toyota.csv",na_values=["????","??"],index_col=0)

print(df.head()) 
print(df.info())
print(df.isna().sum())

df.dropna(inplace=True)





num_data = df.select_dtypes(include=["int64","float64"])

corr = num_data.corr()

sns.heatmap(corr,cmap="crest",annot=True)

plt.show()