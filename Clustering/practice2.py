import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
crop_df = pd.read_csv("DataSets/crop_recommendation.csv")

# Basic data exploration
print(crop_df.info())
print(crop_df.describe())
print(crop_df.isna().sum())
print(crop_df.columns)
print(crop_df.head())
print(crop_df.duplicated().sum())

# Scatter plot temperature vs humidity colored by crop label
sns.scatterplot(data=crop_df, x="temperature", y="humidity", hue="label")
plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0, loc="upper left")
plt.tight_layout()
plt.show()

# Select numeric columns for clustering
numeric_cols = crop_df.select_dtypes(include="number").columns

# Scale numeric features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(crop_df[numeric_cols])

# Convert scaled data to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)

# Initialize and fit KMeans clustering with 3 clusters
kmeans_model = KMeans(n_clusters=23, random_state=42, n_init=25)
scaled_df["cluster"] = kmeans_model.fit_predict(scaled_df)

print(scaled_df.head(10))
print(scaled_df["cluster"].value_counts())

# Calculate silhouette score for 3 clusters
cluster_labels = kmeans_model.labels_
silhouette_avg = silhouette_score(scaled_df.drop(columns=["cluster"]), cluster_labels)
print("Silhouette Score:", silhouette_avg)

# Evaluate silhouette scores and inertia for a range of cluster numbers
cluster_range = range(2,30)
silhouette_scores = []
inertia_values = []

for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=25)
    cluster_preds = kmeans.fit_predict(scaled_df)
    
    inertia_values.append(kmeans.inertia_)
    score = silhouette_score(scaled_df, cluster_preds, sample_size=500, random_state=42)
    silhouette_scores.append(score)
    print(f"Clusters: {num_clusters}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {score:.3f}")

# Plot inertia and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cluster_range, inertia_values, marker="o")
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Inertia')
axes[0].grid(True)

axes[1].plot(cluster_range, silhouette_scores, marker="o")
axes[1].set_title('Silhouette Scores')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True)

plt.tight_layout()
plt.show()
