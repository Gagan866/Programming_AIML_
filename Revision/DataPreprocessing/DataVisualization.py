import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Revision/Data/sales_data.csv")

# print(df)
# print(df.info())
# print(df.describe())
# print(df.isna().sum())
# print(df.duplicated().sum())

# df[["Month","Region","Product_Category"]] = df[["Month","Region","Product_Category"]].astype("category")
# print(df.info())

# sns.barplot(data=df, x="Region", y="Sales")
# plt.title("Highest Sales by Region")
# plt.tight_layout()    
# plt.show()

# profit_sales1 = df.groupby("Product_Category")[["Sales","Profit"]].sum()
# profit_sales1.plot(kind="bar")
# plt.title("Sales vs Profit by Product Category")
# plt.ylabel("Amount")
# plt.tight_layout()
# plt.show()

# sns.lineplot(data=df, x="Month", y="Sales", hue="Region", marker="o")
# plt.title("Monthly Sales Trend by Region")
# plt.tight_layout()
# plt.show()

# sns.barplot(data=df, x="Month", y="Sales", hue="Product_Category", ci=None)
# plt.title("Product Cat and Monthly sales")
# plt.tight_layout()
# plt.show()

# sns.barplot(data=df,x="Region",y="Sales",estimator="sum")
# plt.title("Region total anual sales")
# plt.tight_layout()
# plt.show()

# profit_sales2 = df.groupby("Month")[["Sales","Profit"]].sum()
# profit_sales2.plot(kind="bar")
# plt.title("Monthly Sales and Profit")
# plt.ylabel("Amount")
# plt.tight_layout()
# plt.show()

# plt.scatter(df["Sales"], df["Profit"], alpha=0.6)
# plt.title("Sales vs Profit Scatter Plot")
# plt.xlabel("Sales")
# plt.ylabel("Profit")
# plt.tight_layout()
# plt.show()

# df["Margin"] = df["Profit"]/df["Sales"]*100
# sns.boxplot(df,x="Product_Category",y="Margin")
# plt.tight_layout()
# plt.show()

# top3 = df.groupby("Region",as_index=False)["Profit"].sum().sort_values("Profit",ascending=False).head(3)
# sns.barplot(data=top3,x="Profit", y="Region")
# plt.title("Top 3 Regions by Profit")
# plt.tight_layout()
# plt.show()

# encode = {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3}
# df["Product_Category_Encoded"] = df["Product_Category"].map(encode)

# corr = df[["Sales", "Profit", "Product_Category_Encoded"]].corr()
# print(corr)

# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation between Sales, Profit, and Product Category")
# plt.show()

df["Month"] = pd.Categorical(df["Month"],
    categories=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    ordered=True)
g = sns.FacetGrid(df, col="Region")
g.map_dataframe(sns.lineplot, x="Month", y="Sales", marker="o")
plt.show()

# sns.lineplot(df,x="Month",y="Profit",hue="Region",estimator="mean")
# plt.show()

# sns.barplot(df,x="Product_Category",y="Sales",estimator="sum")
# plt.show()

# profit_pie = df.groupby("Product_Category")["Profit"].sum()
# plt.pie(profit_pie,labels=profit_pie.index ,autopct="%1.1f%%", startangle=90)
# plt.show()

# sns.barplot(data=df, x="Product_Category", y="Sales", estimator=sum, ci=None)
# sns.lineplot(data=df, x="Product_Category", y="Profit", estimator="mean", marker="o", color="red")
# plt.title("Sales (Bar) and Average Profit (Line) by Product Category")
# plt.tight_layout()
# plt.show()


