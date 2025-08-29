import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# df = sns.load_dataset("titanic")

df = pd.read_csv("DataSets/toyota.csv")

df.dropna(inplace=True)

print(df.head(10))

# sns.regplot(x="age",y="fare",data=df,marker="*",fit_reg=True,scatter=True,ci=95,line_kws={"linewidth":2,"color":"black"},scatter_kws={"alpha":0.5})

sns.regplot(x=df["Age"],y=df["Price"],marker="*",fit_reg=True,ci=95,)

plt.grid(True)

sns.set_style("darkgrid", {"grid.color": ".6", "axes.edgecolor": "black"})

sns.set_context("talk")

plt.show()