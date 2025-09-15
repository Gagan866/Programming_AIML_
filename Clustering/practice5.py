# 📦 Import libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

iris = sns.load_dataset("iris")
X = iris.drop("species", axis=1)   # drop labels, keep only numeric features

print("Dataset Shape:", X.shape)
print("First 5 rows:\n", X.head())

# ================================
# 2. Scale Features
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3,max_iter=300,random_state=42,n_init=10)

y_pred = kmeans.fit_predict(X_scaled)

print(kmeans.inertia_)
print(silhouette_score(X_scaled,y_pred))


c_range = range(2,10)
iner = []
silo = []

for i in c_range:
    kmeans1 =KMeans(n_clusters=i,n_init=10,max_iter=300,random_state=42)
    y_pred1 = kmeans1.fit_predict(X_scaled)
    print(kmeans1.inertia_)
    iner.append(kmeans1.inertia_)
    print(silhouette_score(X_scaled,y_pred1))
    silo.append(silhouette_score(X_scaled,y_pred1))
    
print(iner)    
print(silo)    