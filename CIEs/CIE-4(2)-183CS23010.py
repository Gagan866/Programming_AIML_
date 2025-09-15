import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("CIEs/customer_segmentation.csv")

print("_"*40)
print(" "*10,"EDA")
print("_"*40)
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.columns)
print("_"*40)

print("_"*40)
print(" "*10,"Drop CustomerID")
print("_"*40)
df_mod = df.drop("CustomerID",axis=1)
print(df_mod.info())
print("_"*40)

sns.boxplot(df_mod)
plt.show()

scaler = StandardScaler()
df_mod[["Age","Annual Income (k$)","Spending Score (1-100)"]] = scaler.fit_transform(df_mod[["Age","Annual Income (k$)","Spending Score (1-100)"]])
print(df_mod.head(10))

c_range = range(2,20)
inertia = []
silhouette = []

for i in c_range:
    kmeans = KMeans(n_clusters=i,max_iter=300,random_state=42,n_init=10)
    y_pred = kmeans.fit_predict(df_mod)
    inertia.append(kmeans.inertia_)
    score = silhouette_score(df_mod, y_pred)
    silhouette.append(score)
    
    print(f"Clusters: {i}, Inertia: {kmeans.inertia_}, Silhouette Score: {score:.3f}")
    

print("_"*40)
print(" "*10,"Inertia")
print("_"*40)
print(inertia)
print("_"*40)
print("_"*40)
print(" "*10,"Silhouete")
print("_"*40)
print(silhouette)
print("_"*40)

plt.plot(c_range, inertia, marker="o")
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(c_range, silhouette, marker="o")
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)  
plt.tight_layout()
plt.show()

