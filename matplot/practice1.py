import matplotlib.pyplot as plt
import pandas as pd
# data = {
#     'Department': ['IT', 'IT', 'HR', 'Finance', 'HR', 'Finance', 'IT'],
#     'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M'],
#     'Salary': [60000, 65000, 52000, 72000, 50000, 71000, 61000],
#     'Experience': [4, 3, 2, 7, 1, 6, 5]
# }
# df = pd.DataFrame(data)

# plt.scatter(df["Experience"],df["Salary"])

# plt.show()

df = pd.read_csv("DataSets/Titanic.csv",na_values=["??","????"])
drop=df.dropna()
print(drop.head(10))
# plt.scatter(df["HP"],df["Price"],s=100,alpha=0.5,marker="*",edgecolors="black")
# plt.title("Scatter plot")
# plt.xlabel("HP")
# plt.ylabel("Price")
# plt.legend("HI")
# plt.show()

# plt.scatter(df["Price"],df["Age"],c="black",s=10)

# plt.grid(True,linestyle="--",linewidth=2.0)

# plt.xlim(0,40000)

# # plt.ylim(0,10)

# plt.xticks(range(0,40000,5000))

# # plt.yticks(range(1,10,2))

# plt.show()
index = 3
plt.bar(index)