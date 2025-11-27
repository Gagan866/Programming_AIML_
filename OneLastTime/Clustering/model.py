import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("clustering_data.csv",index_col=0)

print(df)

sns.boxplot(df)
plt.show()

sc = StandardScaler()

df[["Age" , "AnnualIncome" , "SpendingScore" , "Tenure" , "Transactions"]] = sc.fit_transform(df[["Age" , "AnnualIncome" , "SpendingScore" , "Tenure" , "Transactions"]])

print(df)

silo = []
inertai = []

range_ = range(2,15)

for i in range_:
    k = KMeans(n_clusters=i,n_init=25,random_state=42)
    ypred = k.fit_predict(df)
    inertai.append(k.inertia_)
    silo.append(silhouette_score(df,ypred))

plt.plot(range_,silo)
plt.show()
plt.plot(range_,inertai)
plt.show()