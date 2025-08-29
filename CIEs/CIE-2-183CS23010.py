import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_csv("CIEs/mobile_sales_data.csv")
missing_rows  = df[df.isna().any(axis=1)]

print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(missing_rows)
print(df.duplicated())


# Handling null values 
df["Brand"] = df["Brand"].astype("category")
df["Store"] = df["Store"].astype("category")
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

df["Brand"] = df["Brand"].fillna(df["Brand"].mode()[0])
df["Store"] = df["Store"].fillna(df["Store"].mode()[0])
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["Units_Sold"] = df["Units_Sold"].fillna(df["Units_Sold"].mean())


print(df.isna().sum())
print(df.info())


# Detecting Outliers
df[["z_Price","z_Units_Sold","z_Total_Sales"]] = zscore(df[["Price","Units_Sold","Total_Sales"]])
print(df.head(10))
outliers = df[df["z_Price"].abs()>3]
outliers = df[df["z_Units_Sold"].abs()>3]
outliers = df[df["z_Total_Sales"].abs()>3]

print(outliers.shape)
print(outliers.head())

# EDA
sns.histplot(df["Price"])
plt.show()
sns.boxplot(df["Price"])
plt.show()


print(df.describe())
# print(pd.pivot_table(df,index="Brand",values="Brand",aggfunc="sum"))
# print(df["Brand"].value_counts())
# print(df.info())

# print("_"*20)
# print("Price_Mean",df["Price"].mean())
# print("Units_Sold_Mean",df["Units_Sold"].mean())
# print("Total_Sales_Mean",df["Total_Sales"].mean())
# print("_"*20)

# print("_"*20)
# print("Price_Median",df["Price"].median())
# print("Units_Sold_Median",df["Units_Sold"].median())
# print("Total_Sales_Median",df["Total_Sales"].median())
# print("_"*20)

# print("Price_mode",df["Price"].mode())
# print("Units_Sold_mode",df["Units_Sold"].mode())
# print("Total_Sales_mode",df["Total_Sales"].mode())

# print("_"*20)
# print("Price_std",df["Price"].std())
# print("Units_Sold_std",df["Units_Sold"].std())
# print("Total_Sales_std",df["Total_Sales"].std())
# print("_"*20)

# print("Price_min",df["Price"].min())
# print("Units_Sold_min",df["Units_Sold"].min())
# print("Total_Sales_min",df["Total_Sales"].min())
# print("_"*20)

# print("Price_max",df["Price"].max())
# print("Units_Sold_max",df["Units_Sold"].max())
# print("Total_Sales_max",df["Total_Sales"].max())
# print("_"*20)

sns.scatterplot(data=df,x="Units_Sold",y="Price")
plt.show()

x = df.groupby("Brand")["Price"].mean()
print(x)
sns.barplot(data=x)
plt.show()

sns.boxplot(data=df,x="Brand",y="Price")
plt.show()
