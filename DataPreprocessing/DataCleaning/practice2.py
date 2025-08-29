import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# df = pd.DataFrame({"Temp":[-10,22,23,24,25,22,23,22,13,14,15,12,13,40,15,16]})

# print(df)

# sns.boxplot(df)

# plt.show()

# q1 = df["Temp"].quantile(0.25)
# q3 = df["Temp"].quantile(0.75)
# IQR = q3-q1

# print("Q1",q1)
# print("Q3",q3)
# print("IQR",IQR)

# l = q1-1.5*IQR
# u = q3+1.5*IQR

# print("L",l)
# print("U",u)

# out = df[(df["Temp"]<l)|(df["Temp"]>u)]

# print(out)




# df = pd.read_csv("DataSets/titanic.csv")

# print(df.info())
# df.dropna(inplace=True)

# sns.boxplot(df["Age"])

# plt.show()

# q1 = df["Age"].quantile(0.25)
# q3 = df["Age"].quantile(0.75)
# IQR = q3-q1

# print("Q1",q1)
# print("Q3",q3)
# print("IQR",IQR)

# l = q1-1.5*IQR
# u = q3+1.5*IQR

# print("L",l)
# print("U",u)

# out = df[(df["Age"]<l)|(df["Age"]>u)]

# print(out)




# df = pd.read_csv("DataSets/titanic.csv")

# print(df.info())

# df.dropna(inplace=True)

# sns.boxplot(df["Fare"])

# plt.show()

# q1 = df["Fare"].quantile(0.25)
# q3 = df["Fare"].quantile(0.75)
# IQR = q3-q1

# print("Q1",q1)
# print("Q3",q3)
# print("IQR",IQR)

# l = q1-1.5*IQR
# u = q3+1.5*IQR

# print("L",l)
# print("U",u)

# out = df[(df["Fare"]<l)|(df["Fare"]>u)]

# print(out)

