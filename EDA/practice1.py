import pandas as pd

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns


df = pd.read_csv("DataSets/Transactions.csv")

print(df.head(10))

print(df.info())

# print(df.isna().sum())

print(df["date"].value_counts())

# sns.countplot(data=df,x=df["category"].sort_values())
# plt.show()
# sns.countplot(data=df,x=df["product_name"].sort_values())
# plt.show()
# sns.countplot(data=df,x="date")
# plt.show()
# sns.barplot(data=df,x="product_name",y="quantity")
# plt.show()
# sns.barplot(data=df,x="category",y="quantity")
# plt.show()
# sns.barplot(data=df,x="product_name",y="price")
# plt.show()
# sns.barplot(data=df,x="category",y="price")
# plt.show()
# sns.lineplot(data=df,x="product_name",y="price")
# plt.show()
# sns.lineplot(data=df,x="category",y="price")
# plt.show()
# sns.barplot(data=df,x="customer_id",y="price")
# plt.show()
sns.boxplot(data=df,x=df["category"].value_counts())
plt.show()

quantity_avg = df["quantity"].mean()
price_avg = df["price"].mean()
quantity_med = df["quantity"].median()
price_med = df["price"].median()
quantity_mode = df["quantity"].mode()
price_mode = df["price"].mode()
print("Quantity mean",quantity_avg)
print("Price mean",price_avg)
print("Quantity median",quantity_med)
print("Price median",price_med)
print("Quantity mode",quantity_mode)
print("Price mode",price_mode)