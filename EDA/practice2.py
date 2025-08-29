import pandas as pd

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns


df = pd.read_csv("DataSets/data_loan.csv")

print(df.head())
print(df.describe())
print(df.info())

# sns.histplot(data=df,x="Loan_type",kde=True)
# plt.show()
# sns.histplot(data=df,x="Gender",kde=True)
# plt.show()
# sns.histplot(data=df,x="Age",kde=True)
# plt.show()
# sns.histplot(data=df,x="Degree",kde=True)
# plt.show()
# sns.histplot(data=df,x="Income",kde=True)
# plt.show()
# sns.histplot(data=df,x="Credit_score",kde=True)
# plt.show()
# sns.histplot(data=df,x="Loan_length",kde=True)
# plt.show()
# sns.histplot(data=df,x="Signers",kde=True)
# plt.show()
# sns.histplot(data=df,x="Citizenship",kde=True)
# plt.show()

# sns.histplot(data=df,x="Age",y="Loan_type")
# plt.show()
# sns.boxplot(data=df,x="Default",y="Age")
# plt.show()
# sns.barplot(data=df,x="Loan_type",y="Credit_score")
# plt.show()
# sns.histplot(data=df,x="Credit_score",y="Income")
# plt.show()
# sns.boxplot(data=df,x="Default",y="Income")
# plt.show()
sns.barplot(data=df,x="Gender",y="Default")
plt.show()