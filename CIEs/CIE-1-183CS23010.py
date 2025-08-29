import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CIEs/cie1salesdata.csv")

print(df.head())
print(df.info())
print(df.describe())

# Total Sales by region 
print("__"*20)
s = pd.pivot_table(df,index="Region",values="Sales",aggfunc="sum",sort=False)
print("Total Sales by region ")
print(s)
sns.barplot(data=df,x="Region",y="Sales",palette="deep")
plt.title("Total sales by region")
plt.show()
print("__"*20)

# # Monthly sales across all products
print("__"*20)
sns.barplot(data=df,x="Product",y="Sales",palette="deep",hue="Month")
plt.title("Monthly sales across all products")
plt.show()
print("__"*20)

# # Percentage of sales by category
print("__"*20)
pr = df.groupby("Category")["Sales"].agg(sum)
print(pr)
plt.pie(x=pr,labels=df["Category"].unique(),autopct="%.f%%")
plt.title("Percentage of sales by category")
plt.show()
print("__"*20)

# Total Sales Region Category
print("__"*20)
sns.barplot(data=df,x="Region",y="Sales",palette="deep",hue="Category")
plt.title("Total Sales by Region and Category")
plt.show()
print("__"*20)

# Relation ship between Sales and Quantity sold
print("__"*20)
sns.scatterplot(data=df,x="Quantity",y="Sales",palette="deep")
plt.title("Relation ship between Sales and Quantity sold")
plt.show()
sns.regplot(data=df,x="Quantity",y="Sales")
plt.title("Relation ship between Sales and Quantity sold")
plt.show()
print("__"*20)

# Pivot table total sales per region 
print("__"*20)
sales = pd.pivot_table(df,index="Region",values="Sales",aggfunc="sum")
print("Pivot table total sales per region ")
print(sales)
print("__"*20)

# # Pivot table Total quantity for each product
print("__"*20)
quan =pd.pivot_table(df,index="Product",values="Quantity",aggfunc="sum")
print("Pivot table for Total quantity for each product")
print(quan)
print("__"*20)

# Highest total sales per region
print("__"*20)
hig_r = pd.pivot_table(df,index="Region",values="Sales",aggfunc={"Sales":"sum"}).sort_values(by="Sales",ascending=0)
print("Highest total sales per region")
print(hig_r[:1])
print("Highest sales ",hig_r.max())
sns.barplot(data=df,x="Region",y="Sales",palette="deep")
plt.title("Highest total sales per region")
plt.show()
print("__"*20)

# Max Total sales 
print("__"*20)
hig_m = pd.pivot_table(df,index="Month",values="Sales",aggfunc="sum").sort_values(by="Sales",ascending=0)
print(" Max Total sales month rec")
print(hig_m)
print(hig_m[:1])
print(hig_m.max())
sns.barplot(data=df,x="Month",y="Sales",palette="deep")
plt.show()
print("__"*20)

# Pivot table with region as r and cat as c showing sum of sales 
print("__"*20)
pvt = pd.pivot_table(df,index="Region",columns="Category",values="Sales",aggfunc="sum")
print("Pivot table with region as r and cat as c showing sum of sales ")
print(pvt)
print("__"*20)

# Top 3 products
print("__"*20)
top3=pd.pivot_table(df,index="Product",columns="Region",values="Sales",aggfunc="sum",sort=False)
print("Top 3 products")
print(top3[:3])
print("__"*20)

# Avg sales
print("__"*20)
avg_sales = pd.pivot_table(df,index="Category",values="Sales",aggfunc="mean")
print("Avg sales")
print(avg_sales)
print("__"*20)