import pandas as pd

df = pd.read_csv("DataSets/titanic.csv")

# print(df.head())

# print(df["Embarked"].unique())

# df["Embarked"] = df["Embarked"].astype("category").cat.codes

# print(df.head())
# print(df["Embarked"].unique())

# Check the mode (most frequent value) of the Embarked column




# Find mode value (scalar)
mode_embarked = df['Embarked'].mode()[0]
print("Mode of Embarked:", mode_embarked)

# Fill missing values without chained assignment warning:
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# Verify no missing values remain
print("Missing values in Embarked after fill:", df['Embarked'].isna().sum())
