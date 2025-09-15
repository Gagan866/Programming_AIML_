import pandas as pd

order = pd.read_csv("DataSets/orders.csv")
product = pd.read_csv("DataSets/products.csv")

combine = pd.merge(order,product,on="ProductID",how="",)

combine["Total_Price"] = combine["Price"]*combine["Quantity"]

print(combine)