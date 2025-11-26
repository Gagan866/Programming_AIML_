import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Revision/Data/clustering_practice_dataset.csv")

print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.columns)

df.drop_duplicates(inplace=True)

df[["AnnualIncome","Experience"]] = df[["Experience","AnnualIncome"]].fillna(df[["Experience","AnnualIncome"]].mean())

print(df.isna().sum())

df = pd.get_dummies(df,columns=["City"],drop_first=True)

print(df)

num_col = ['Age', 'AnnualIncome', 'Experience', 'HoursWorked']

sc = StandardScaler()

df[num_col] = sc.fit_transform(df[num_col])

inertia = []
silo = []

range_ = range(3,10)

for i in range_:
    k = KMeans(n_clusters=i)
    clu = k.fit_predict(df)
    inertia.append(k.inertia_)
    silo.append(silhouette_score(df,clu))

plt.plot(range_,silo)
plt.show()
plt.plot(range_,inertia)
plt.show()
