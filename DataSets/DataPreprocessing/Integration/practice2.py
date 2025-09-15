import pandas as pd

users = pd.read_csv("DataSets/Bank/users.csv")
transactions = pd.read_csv("DataSets/Bank/transactions.csv")
logins = pd.read_csv("DataSets/Bank/logins.csv")

combine = pd.merge(users,logins,on="user_id",how="inner")

combine2 = pd.merge(combine,transactions,on="user_id",how="left")

print(combine)
print(combine2)