import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import zscore

# df = pd.DataFrame({"Temp":[-10,22,23,24,25,22,23,22,13,14,15,12,13,40,15,16]})

# df["z_score"] = zscore(df['Temp'])

# out = df[df["z_score"].abs()>1]
# print(df["z_score"])
# print(out)

df = pd.read_csv("DataSets/titanic.csv")

df.dropna(inplace=True)

q1 = df["Fare"].quantile(0.25)
q3 = df["Fare"].quantile(0.75)
IQR = q3-q1

print("Q1",q1)
print("Q3",q3)
print("IQR",IQR)

l = q1-1.5*IQR
u = q3+1.5*IQR

print("L",l)
print("U",u)

out = df[(df["Fare"]<l)|(df["Fare"]>u)]

print(out)

df["z_score"] = zscore(df['Fare'])

out1 = df[df["z_score"].abs()>2]
print(df["z_score"])
print(out1)




