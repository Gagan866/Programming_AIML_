import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")

print(df.head(10))

print(df.info())

print(df.describe())
df.dropna(inplace=True)
plt.bar(df["class"].sort_values().unique(),df["class"].value_counts())

plt.show()