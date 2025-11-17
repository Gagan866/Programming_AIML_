import pandas as pd

df = pd.read_csv("Revision\Data\student_scores.csv")
print(df)

print(pd.pivot_table(df,index="GENDER",aggfunc="mean"))
print(pd.pivot_table(df,index="GENDER",aggfunc="min"))
print(pd.pivot_table(df,index="GENDER",aggfunc="max")) 
print(pd.pivot_table(df,index="GENDER",aggfunc="count")) 
print(pd.pivot_table(df,index="GENDER",aggfunc="sum")) 


df["Total"] = df["DA"] + df["CIE"] + df["ASSNT"]

print(df)

df["DA_Per"] = df["DA"] *100 / 20
df["CIE_Per"] = df["CIE"] *100 / 150
df["ASSNT_Per"] = df["ASSNT"] *100 / 10
df["Total_Per"] = df["Total"] *100 / 240

print(df)

df["DA_Range"] = pd.cut(df["DA_Per"], bins=[0, 50, 70, 85, 100], labels=["Poor", "Average", "Good", "Excellent"])
print(pd.pivot_table(df, index="GENDER", columns="DA_Range", values="DA", aggfunc="count", fill_value=0,margins=True))
