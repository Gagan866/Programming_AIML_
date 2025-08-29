import pandas as pd

df = pd.read_csv("C:/183CS23010/Sem 5/AI&ML/Programming/data/Toyota.csv",index_col=0,na_values=["??","????"])
print(df.shape)
print(df.dropna())
# print(df.info())
# fueltype = df.groupby("FuelType")["FuelType"].count()
# print(fueltype)

# print(df.groupby(['Automatic','FuelType']).size())

# # df['Transmission'] = df['Automatic'].map({1: 'Automatic', 0: 'Manual'})
# # print(df.groupby(['FuelType', 'Transmission']).size().unstack(fill_value=0))
# print(df.loc[:10,'Price'])
# print(df[["FuelType","Price"]])
# print(df.iloc[:10,[0,2]])
# print(df[(df["Price"]>10000)&(df["FuelType"].isin(["Petrol","Diesel"]))].head(10))

# print(df[df["FuelType"].isna()])


