import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

scaler = MinMaxScaler()
standard = StandardScaler()


# df = pd.read_csv("DataSets/data_loan.csv")

# sns.histplot(df,x="Income",kde=True)
# plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()


# df[["Age","Income"]] = scaler.fit_transform(df[["Age","Income"]])
# print(df[["Age","Income"]])

# sns.histplot(df,x="Income",kde=True)
# plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()


# df[["Age","Income"]] = standard.fit_transform(df[["Age","Income"]])
# print(df[["Age","Income"]])

# sns.histplot(df,x="Income",kde=True)
# plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()

df = pd.read_csv("DataSets/toyota.csv")

# print(df.info())

# sns.histplot(df,x="Price",kde=True)
# plt.show()
sns.histplot(df,x="Age",kde=True)
plt.show()

# df[["Age","Price"]] = scaler.fit_transform(df[["Age","Price"]])
# print(df[["Age","Price"]])

# # sns.histplot(df,x="Price",kde=True)
# # plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()

df[["Age","Price"]] = standard.fit_transform(df[["Age","Price"]])
print(df[["Age","Price"]])

# sns.histplot(df,x="Price",kde=True)
# plt.show()
sns.histplot(df,x="Age",kde=True)
plt.show()