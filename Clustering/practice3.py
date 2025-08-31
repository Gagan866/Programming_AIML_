import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score

# Load dataset
# df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/crop_recommendation.csv")
df = pd.read_csv("DataSets/crop_recommendation.csv")


# Display basic info and stats
print(df.head(10))
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isna().sum())
print(df.columns)

# Correlation matrix (optional to plot)
corr = df.corr(numeric_only=True)
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()

# Select numeric columns and scale data
numeric_cols = df.select_dtypes(include=["int64","float64"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_cols)
df_scaled = pd.DataFrame(scaled_features, columns=numeric_cols.columns)

# Apply initial KMeans clustering with 23 clusters
kmeans = KMeans(n_clusters=23, random_state=42, n_init=25)
df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

print(df_scaled["cluster"].value_counts())
print(df_scaled["cluster"].head())

# Cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)

# Silhouette score for 23 clusters
sil_score = silhouette_score(df_scaled, labels=labels)
print("Silhouette Score:", sil_score)
print(df["label"].unique())

# Evaluate silhouette scores and inertia for a range of cluster numbers
k_values = range(2, 30)
silhouette_scores = []
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
    cluster_labels = kmeans.fit_predict(df_scaled)
    
    inertias.append(kmeans.inertia_)
    score = silhouette_score(df_scaled, labels=cluster_labels, sample_size=500, random_state=42)
    silhouette_scores.append(score)
    print(f"k = {k}, inertia = {kmeans.inertia_:.2f}, silhouette_score = {score:.3f}")

# Plot inertia and silhouette scores side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Elbow (Inertia) plot
axs[0].plot(k_values, inertias, marker="o")
axs[0].set_xlabel("Number of Clusters (k)")
axs[0].set_ylabel("Inertia")
axs[0].set_title("Elbow Method (Inertia vs. k)")
axs[0].grid(True)

# Silhouette score plot
axs[1].plot(k_values, silhouette_scores, marker="o", color='orange')
axs[1].set_xlabel("Number of Clusters (k)")
axs[1].set_ylabel("Silhouette Score")
axs[1].set_title("Silhouette Score vs. Number of Clusters (k)")
axs[1].grid(True)

plt.tight_layout()
plt.show()
