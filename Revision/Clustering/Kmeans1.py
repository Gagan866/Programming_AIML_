import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

df = pd.read_csv("Revision/Data/kmeans1.csv",index_col=0)

print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())

# sns.boxplot(df)
# plt.show()

scl = StandardScaler()

df_scaled = scl.fit_transform(df)

print(df_scaled)

df_1 = pd.DataFrame(df_scaled,columns=df.columns)

print(df_1)

kmeans = KMeans(n_clusters=5,n_init=25,random_state=42)
kmeans.fit(df_1)

print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

cluster_lables = kmeans.labels_

print(cluster_lables)

silo = silhouette_score(df_1,labels=cluster_lables)

print(silo)


silos = []
range_ = range(2,10)
inertia = []

for i in range_:

    km = KMeans(n_clusters=i,random_state=42,n_init=25)
    labels = km.fit_predict(df_1)

    s = silhouette_score(df_1,labels=labels)
    silos.append(s)

    i = km.inertia_
    inertia.append(i)

print(silos)
print(inertia)

plt.plot(range_,silos)
plt.show()

plt.plot(range_,inertia)
plt.show()

