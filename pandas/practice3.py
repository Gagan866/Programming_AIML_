import pandas as pd 

df = pd.read_csv("DataSets/Toyota.csv")

print(df.groupby(["FuelType","Age"])["Price"].count().head(20))