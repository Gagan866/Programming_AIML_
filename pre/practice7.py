import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("DataSets/data.csv")

# Numerical - Histogram
sns.histplot(df['CGPA'], kde=True)
plt.title("CGPA Distribution")
plt.show()

# # Numerical - Boxplot
# sns.boxplot(x=df['CGPA'])
# plt.title("CGPA Boxplot")
# plt.show()

# # Categorical - Countplot
# sns.countplot(x='Branch', data=df)
# plt.title("Student Count by Branch")
# plt.show()

# # Descriptive Stats
# print(df['CGPA'].describe())  # mean, std, min, 25%, 50%, 75%, max
# print(df['Branch'].value_counts())
