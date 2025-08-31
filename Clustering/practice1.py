import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


df = pd.read_csv("DataSets/crop_recommendation.csv")

print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.columns)
print(df.head())
print(df.duplicated().sum())


sns.scatterplot(data=df,x=df["temperature"],y=df["humidity"],hue="label")
plt.legend(bbox_to_anchor=(1.05,1),borderaxespad=0,loc="upper left")
plt.tight_layout()
plt.show()   #for all relation


# sns.pairplot(data=df,hue="label")
# plt.show()


# sns.heatmap(data=(df.corr(numeric_only=True)),cmap="crest",annot=True)
# plt.show()


# Select numeric columns
numeric_columns = df.select_dtypes(include="number").columns

# Scale numeric columns
scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[numeric_columns])

# Convert scaled array back to DataFrame
df_num1 = pd.DataFrame(scaled_values, columns=numeric_columns)

# Assign clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=25)
df_num1["cluster"] = kmeans.fit_predict(df_num1)

print(df_num1.head(10))
print(df_num1["cluster"].value_counts())

# Compute silhouette score using only the feature columns
labels = kmeans.labels_
sil_score = silhouette_score(df_num1.drop(columns=["cluster"]), labels)
print("Score:", sil_score)


k_values = range(22,50)
scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=25)
    labels = kmeans.fit_predict(df_num1)
    
    score = silhouette_score(df_num1,labels,sample_size=500,random_state=42)
    scores.append(score)
    print(f"k={k} , Silo = {score:.3f}")
    
   
plt.figure(figsize=(8,5))
plt.plot(k_values,scores,marker="o")
plt.xticks(k_values)
plt.xlabel("No of cluster ")
plt.ylabel("Silo")
plt.title("Silo")
plt.grid(True)
plt.show()    




k_values = range(22,50)
sil_scores = []
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
    labels = kmeans.fit_predict(df_num1)
    
    inertias.append(kmeans.inertia_)
    
    score = silhouette_score(df_num1,labels,sample_size=500,random_state=42)
    sil_scores.append(score)
    print(f"k = {k},inertia = {kmeans.inertia_:.2f,silo = {score:.3f}}")
    
    
fig,ax = plt.subplots(1,2,figsize={14,5})

ax[0].plot(k_values,inertias,marker="o")
ax[0].grid(True)

ax[1].plot(k_values,sil_score,marker="o")    
ax[1].grid(True)

plt.tight_layout()
plt.show()