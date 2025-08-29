import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x = [10, 20, 30, 40]
y = [5, 15, 25, 35]


# Figure with Axes

# fig = plt.figure(figsize=(10,4))

# ax = fig.add_axes([0.1,0.1,0.8,0.8])
# ax1 = fig.add_axes([0.5,0.5,0.3,0.3])

# ax.plot(x,y)
# ax1.plot(x,y)

# plt.show()


# Subplots 

# fig, ax = plt.subplots(2,2 , figsize=(10,4))

# ax[0,0].plot([10, 20, 30, 40],[5, 15, 25, 35])
# ax[0,0].set_title("HI")

# ax[0,1].bar(x,y)
# ax[0,1].set_title("HI")

# ax[1,0].hist(x,y)
# ax[1,0].set_title("HI")

# ax[1,1].scatter(x,y)
# ax[1,1].set_title("HI")

# plt.tight_layout()

# plt.show()


# import numpy as np

# categories = ['Q1', 'Q2', 'Q3', 'Q4']
# product_a = [100, 150, 200, 180]
# product_b = [90, 140, 170, 160]

# x = np.arange(len(categories))
# width = 0.35

# plt.bar(x - width/2, product_a, width, label='Product A', color='blue')
# plt.bar(x + width/2, product_b, width, label='Product B', color='green')

# plt.xticks(x, categories)
# plt.xlabel("Quarter")
# plt.ylabel("Sales")
# plt.title("Quarterly Sales Comparison")
# plt.legend()
# plt.show()


# tips = sns.load_dataset("tips")
# sns.countplot(x='day', data=tips)
# plt.title("Number of Tips per Day")
# plt.show()


# data = pd.DataFrame({
#     "Dept": ['Edu', 'Edu', 'Health', 'Health', 'Defence', 'Defence'],
#     "Year": ['2023', '2024', '2023', '2024', '2023', '2024'],
#     "Budget": [20, 25, 30, 28, 22, 27]
# })

# sns.barplot(data=data, x='Dept', y='Budget', hue='Year', palette='muted')
# plt.title("Budget Comparison by Year")
# plt.show()


# df = pd.DataFrame({
#     'Day': [1, 2, 3, 1, 2, 3],
#     'Visitors': [120, 130, 115, 100, 105, 98],
#     'Site': ['A', 'A', 'A', 'B', 'B', 'B']
# })

# sns.lineplot(data=df, x='Day', y='Visitors', hue='Site')
# plt.title("Visitors on Two Sites")
# plt.show()


# df = pd.DataFrame({
#     'Age': [23, 25, 30, 35, 40, 22, 28],
#     'Salary': [30000, 35000, 50000, 60000, 65000, 28000, 45000],
#     'Department': ['IT', 'HR', 'IT', 'Sales', 'Sales', 'HR', 'IT']
# })

# sns.scatterplot(data=df, x='Age', y='Salary', hue='Department', style='Department', s=100)
# plt.title("Salary by Age and Department")
# plt.grid(True)
# plt.show()

flights = sns.load_dataset("flights")
pivot = flights.pivot(index="month", columns="year", values="passengers")

sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Flights Heatmap")
plt.show()
