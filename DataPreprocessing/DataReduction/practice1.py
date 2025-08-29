import pandas as pd

df = pd.read_csv("DataSets/titanic.csv")
sample = df.sample(frac=0.5,random_state=40)

sample.dropna(inplace=True)

df1 = pd.pivot_table(sample,index="Sex",columns="Pclass",values="PassengerId",aggfunc="count")

print(df1)
