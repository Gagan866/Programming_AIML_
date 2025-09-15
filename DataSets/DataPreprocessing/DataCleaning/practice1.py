import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

# df = pd.read_csv("DataSets/titanic.csv")

# print(df.isna().sum())

# print(df.info())

# miss_row = df[df["Age"].isna()].index
# print(miss_row)
# print(miss_row.index)

# fill = df["Age"].fillna(df["Age"].mean(),inplace=True)
# # fill1 = df["Embarked"].fillna(df["Embarked"].mean(),inplace=True)
# print(fill)
# # print(fill1)
# sns.heatmap(data=df.isna(),cbar=True)

# plt.show()


# KNN


# df = pd.DataFrame({"Temp":[-10,None,23,None,25,None,23,22,None,14,15,12,13,40,15,None]})

# imputer = KNNImputer(n_neighbors=3)

# df = imputer.fit_transform(df)

# # print(df_i)
# print(df)






df = pd.read_csv("DataSets/titanic.csv")

print(df.isna().sum())

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_num = df[numeric_cols]

# Apply KNNImputer to ONLY numeric columns
imputer = KNNImputer(n_neighbors=3)
df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num))

# If you want to combine back with non-numeric columns:
df_final = df.copy()
df_final[numeric_cols] = df_num_imputed

print(df.isna().sum())
print(df_final.isna().sum())

