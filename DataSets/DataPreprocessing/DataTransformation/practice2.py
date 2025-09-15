import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# df = pd.read_csv("DataSets/data_loan.csv")

encoder_lab = LabelEncoder()
encoder_hot = OneHotEncoder(drop="first",sparse_output=False)

# df["Degree"].dropna(inplace=True)
# print(df.info())
# print(df["Degree"].unique())

# df["Degree_en"] = encoder_lab.fit_transform(df["Degree"])

# Degree = {"HS      ":0,"College ":1,"Graduate":2}

# df["Degree_en"] = df["Degree"].map(Degree)

# print(df[["Degree","Degree_en"]])



# data = {
#     "Name": [
#         "Rohan", "Priya", "Amit", "Sana", "Vikram", "Sneha", "Rahul", "Meena", "Alok", "Kavya",
#         "Dinesh", "Jaya", "Ritu", "Deepak", "Isha", "Guru", "Pooja", "Anil", "Akash", "Tara"
#     ],
   
#     "PreferredMode": [
#         "Online", "Offline", "Hybrid", "Online", "Offline", "Online", "Hybrid", "Online", "Offline", "Online",
#         "Hybrid", "Offline", "Online", "Hybrid", "Online", "Offline", "Hybrid", "Offline", "Online", "Hybrid"
#     ],
#     "Marks": [
#         85, 77, 90, 82, 70, 88, 79, 91, 76, 89,
#         89, 78, 83, 85, 85, 74, 81, 73, 86, 70
#     ]
# }

# df = pd.DataFrame(data)

# print(df)


# Lable Encoding Auto
# df["PreferredMode_L"] = encoder_lab.fit_transform(df["PreferredMode"])
# print(df[["PreferredMode","PreferredMode_L"]])


# # Lable Encoding Manual
# mode = {"Online":0,"Offline":1,"Hybrid":2}
# df["PreferredMode_M"] = df["PreferredMode"].map(mode)
# print(df[["PreferredMode","PreferredMode_L","PreferredMode_M"]])



# # One Hot Encoding Pandas
# dummies_mode = pd.get_dummies(df,columns=["PreferredMode"],drop_first=True)
# print(dummies_mode)
# print(df)



# # One Hot Encoding Preprocessing
# one_hot = encoder_hot.fit_transform(df[["PreferredMode"]])
# print(one_hot)
# feature_names = encoder_hot.get_feature_names_out(["PreferredMode"])
# print(feature_names)
# data = pd.DataFrame(one_hot,columns=[feature_names])

# df = pd.concat([df,data],axis=1)
# print(df)


####Ranking

# df["Rank_min"] = df["Marks"].rank(method="min")
# print(df.sort_values("Marks"))

# df["Rank_max"] = df["Marks"].rank(method="max")
# print(df.sort_values("Marks"))


# df["Rank_dense"] = df["Marks"].rank(method="dense")
# print(df.sort_values("Marks"))


# df["Rank_first"] = df["Marks"].rank(method="first")
# print(df.sort_values("Marks"))


# df["Rank_average"] = df["Marks"].rank(method="average")
# print(df.sort_values("Marks"))




### Discretization
# ages = pd.Series([21, 25, 32,35, 45, 52, 61, 72, 85,10])
# bins = [20, 30, 40, 50, 60, 70, 80, 90]

# # Create bins and label them
# age_groups = pd.cut(ages, bins=4, labels=["20s", "30s", "40s", "50s"])
# print(age_groups)




# df = pd.read_csv("DataSets/Transactions.csv")

# # print(df.info())
# print(df.head())
# df["date"] = pd.to_datetime(df["date"],errors="coerce",yearfirst=True)
# print(df.head())
# print(df.info())

# df["time"] = pd.to_datetime(df["time"])

# df[["product_name","category"]] = df[["product_name","category"]].astype("category")

# print(df.info())

# print(df.head())


# df["month"] = df["date"].dt.month

# print(df)

# apparel_df = df[df["category"] == "Apparel"]

# print(apparel_df)

# print(pd.pivot_table(df,index="category",columns="month",values="quantity",aggfunc="sum"))

# print(df.groupby(["category","month"])["quantity"].count())

# print(pd.pivot_table(apparel_df,index="month",values="quantity",aggfunc="sum"))

# print(apparel_df.groupby("month")["quantity"].sum())

# # print(df["date"].resample(("m")).mean())
# df = df.set_index("date")

# monthly_sales = df.resample("M").sum()
# print(monthly_sales)


import pandas as pd

# Sample DataFrame
data = {
    "date": ["2023-06-01", "2023-06-05", "2023-07-01", "2023-07-15"],
    "sales": [100, 150, 200, 250],
}
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])

# Make 'date' the index (important for resampling)
df = df.set_index("date")

# Resample by month and sum sales
monthly_sales = df.resample("M").sum()
print(monthly_sales)