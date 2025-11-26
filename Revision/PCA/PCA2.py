import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA

df = pd.read_csv("Revision/Data/pca_practice_dataset.csv")
print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())
print(df.columns)

df.drop_duplicates(inplace=True)

df[['Feature1',"Feature4"]] = df[['Feature1',"Feature4"]].fillna(df[['Feature1',"Feature4"]].mean())

print(df.isna().sum())

sns.boxplot(df)
plt.show()

win_col = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

for i in win_col:
    df[i] = winsorize(df[i])

df = pd.get_dummies(df,columns=["Category"],drop_first=True)

num_features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
sc = StandardScaler()
scaled_data = sc.fit_transform(df[num_features])

range_ = range(1,4)

explained_var_list=[]
for i in range_:
    p = PCA(n_components=i)
    p.fit(scaled_data)
    explained_var_list.append(sum(p.explained_variance_ratio_))
    print(f"Components = {i} --> Cumulative Variance = {sum(p.explained_variance_ratio_):.4f}")

# ---------------------------------------------------------
# PLOT: CUMULATIVE EXPLAINED VARIANCE
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(range_, explained_var_list, marker='o')
plt.title("Cumulative Explained Variance vs Number of Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.axhline(0.95, color='red', linestyle='--', label="95% Threshold")
plt.legend()
plt.show()