
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns


df = pd.read_csv("DataSets/Pricing and Risk.csv")

print(df.head())
print(df.describe())
print(df.info())

print(df.duplicated().sum())
print(df["Claim_Amount"].value_counts())

# sns.histplot(data=df,x="Policy_Type")
# plt.show()
# sns.histplot(data=df,x="Premium_Amount")
# plt.show()
# sns.histplot(data=df,x="Insured_Amount")
# plt.show()
# sns.histplot(data=df,x="Location")
# plt.show()
# sns.histplot(data=df,x="Risk_Score")
# plt.show()

# sns.scatterplot(data=df,x="Insured_Amount",y="Premium_Amount",palette="deep")
# plt.show()

# sns.regplot(data=df,x="Insured_Amount",y="Risk_Score")
# plt.show()

print(df["Premium_Amount"].skew())

neg = np.random.randint(-100,0,100)

# sns.kdeplot(neg)
# plt.show()
df1=pd.DataFrame(neg)
sns.histplot(neg,kde=True)
# plt.xticks(range(-200,0,50))
plt.show()
print(df1.skew())
print(df1.kurtosis())