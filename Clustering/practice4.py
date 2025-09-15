# 📦 Import libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ================================
# 1. Load Dataset
# ================================
iris = sns.load_dataset("iris")
X = iris.drop("species", axis=1)   # drop labels, keep only numeric features

print("Dataset Shape:", X.shape)
print("First 5 rows:\n", X.head())

# ================================
# 2. Scale Features
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 3. Manual KMeans (fixed k=3)
# ================================
kmeans_manual = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=42)
y_manual = kmeans_manual.fit_predict(X_scaled)

# Add cluster labels to dataset
iris["Cluster_manual"] = y_manual

print("\n--- Manual KMeans Results (k=3) ---")
print("Inertia (WCSS):", kmeans_manual.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, y_manual))

# Plot clusters
plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_manual, cmap="viridis", s=50)
plt.scatter(kmeans_manual.cluster_centers_[:, 0], kmeans_manual.cluster_centers_[:, 1],
            s=200, c="red", marker="X", label="Centroids")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.title("KMeans Clustering (Manual k=3)")
plt.legend()
plt.show()

# ================================
# 4. Loop for Multiple k (Elbow + Silhouette)
# ================================
cluster_range = range(2, 11)  # test k=2 to 10
inertia_values = []
silhouette_scores = []

print("\n--- Looping Over k ---")
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_preds = kmeans.fit_predict(X_scaled)
    
    inertia_values.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, cluster_preds)
    silhouette_scores.append(score)
    
    print(f"Clusters: {num_clusters}, Inertia: {kmeans.inertia_}, Silhouette Score: {score:.3f}")

# ================================
# 5. Plots for Evaluation
# ================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow method
axes[0].plot(cluster_range, inertia_values, marker="o")
axes[0].set_title("Elbow Method (Inertia vs k)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].grid(True)

# Silhouette scores
axes[1].plot(cluster_range, silhouette_scores, marker="o")
axes[1].set_title("Silhouette Scores vs k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.show()
