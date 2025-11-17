import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler

df = pd.read_csv("Revision\Data\customer_purchases.csv")

print(df)

print(df.isna().sum())
print(df.isna().mean() * 100)

df[["Gender","Membership_Level","Purchased"]] = df[["Gender","Membership_Level","Purchased"]].astype("category")
print(df.info())

print(df["Gender"].unique())
print(df["Membership_Level"].unique())
print(df["Purchased"].unique())

df["Age"].fillna(df["Age"].mean(),inplace=True)
print(df.isna().sum())

df["Annual_Income (₹)"].fillna(df["Annual_Income (₹)"].median(),inplace=True)
print(df.isna().sum())

df["Membership_Level"].fillna(df["Membership_Level"].mode()[0],inplace=True)
print(df.isna().sum())

df["Spending_Score"] = df["Spending_Score"].interpolate()
print(df.isna().sum())

df["Gender"] = df["Gender"].replace({
                                    "MALE":"Male",
                                    "female":"Female"})
print(df["Gender"].unique())
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print(df["Gender"].unique())

ohe = OneHotEncoder(sparse_output=False,drop="first")
encode = ohe.fit_transform(df[["Membership_Level"]])
mem_col = ohe.get_feature_names_out(["Membership_Level"])
eco_df = pd.DataFrame(encode,columns=mem_col)
df = pd.concat([df,eco_df],axis=1)
print(df)

print(df[["Age","Annual_Income (₹)","Spending_Score"]].describe())
mms = MinMaxScaler()
df[["Age","Annual_Income (₹)","Spending_Score"]] = mms.fit_transform(df[["Age","Annual_Income (₹)","Spending_Score"]])
print(df)
print(df[["Age","Annual_Income (₹)","Spending_Score"]].describe())

print(df.isna().sum())
print(df.head(5))