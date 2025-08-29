import statistics 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random 


sns.set_theme(style="darkgrid")

forbes_data = pd.read_csv("DataSets/forbes1.csv")

print(forbes_data.describe())
print(forbes_data.info())
print(forbes_data.head(10))

print(forbes_data.isnull().sum())

forbes_data.dropna(inplace=True)

# print(forbes_data.isnull().sum())

# mean = forbes_data["Market Value"].mean()
# print("Mean",mean)
# median = forbes_data["Market Value"].median()
# print("Median",median)
# diff_mean_median = mean-median
# print("Difference",diff_mean_median)


# mod = forbes_data["Sector"].mode()
# print("Mode",mod)
# mod1 = forbes_data["Country"].mode()
# print("Mode",mod1)
# mod2 = forbes_data["Industry"].mode()
# print("Mode",mod2)

# val_Count = forbes_data["Sector"].value_counts()
# print("Count",val_Count)
# val_Count1 = forbes_data["Country"].value_counts()
# print("Count",val_Count1)
# val_Count2 = forbes_data["Industry"].value_counts()
# print("Count",val_Count2)

# sns.countplot(data=forbes_data,x="Sector",width=0.5,label=forbes_data["Sector"].unique())
# plt.figure(figsize=(20,10))
# plt.xticks(rotation=90)
# plt.legend()
# plt.tight_layout()
# plt.show()


# q1 = np.quantile(forbes_data["Profits"],.25)
# q3 = np.quantile(forbes_data["Profits"],.75)
# print(q1)
# print(q3)
# print("IQR",q3-q1)

# plt.figure(figsize=(12,6))
# sns.boxplot(data=forbes_data,x="Profits",showmeans=True)
# plt.xlim(-5,10)
# plt.xticks(range(-5,10,1))
# plt.show()


# max = forbes_data["Sales"].max()
# min = forbes_data["Sales"].min()
# print("Range",max-min)

# q1 = np.quantile(forbes_data["Sales"],.25)
# q3 = np.quantile(forbes_data["Sales"],.75)
# print("IQR",q3-q1)

# std = np.std(forbes_data["Sales"])
# var = np.var(forbes_data["Sales"])
# print(std)
# print(std**2)
# print(var)

# def coin_flip(n):
#     out = []
#     for i in range(1,n+1):
#         toss = r.choice(["HHHH", "HHHT", "HHTH", "HTHH", "THHH", "HHTT", "HTHT", "HTTH", 
# "THHT", "THTH", "TTHH", "HTTT", "THTT", "TTHT", "TTTH", "TTTT"
# ])
#         # print(toss)
#         out.append(toss)
#     # print(out)
#     return(out)
    
# def dice(n):
#     out = []
#     for i in range(1,n+1):
#         roll = r.choice([2,3,4,5,6,7,8,9,10,11,12])
#         # print(toss)
#         out.append(roll)
#     # print(out)
#     return(out)
    

# x = coin_flip(1000)
# y = dice(1000)
# sns.countplot(x=x  , palette="deep")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
# sns.countplot(x=y  , palette="deep")
# plt.show()


data= np.random.normal(loc=0,scale=10,size=10000000)
data2= np.random.normal(loc=0,scale=5,size=10000000)

sns.histplot(x=data)
plt.text(x=30,y=20000,s="r$/sigma$")
plt.show()
sns.kdeplot(x=data)
sns.kdeplot(x=data2)
plt.show()