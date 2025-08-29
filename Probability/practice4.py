import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

def sample_mean(age, mean):
   
    for i in range(100):
        x = list(map(int,random.choices(age,k=10)))
        y = np.mean(x)
        mean.append(int(y))
       
def dataset_mean(df, mean1):
    
    for i in range(100):
        x = random.choices(df["Retirement_Age"],k=10)
        y = np.mean(x)
        mean1.append(int(y))
        
age = np.random.randint(30,90,10000)

mean = list()

sample_mean(age, mean)



df = pd.read_csv("DataSets/retirement_age_dataset.csv")

mean1 = [] 

dataset_mean(df, mean1)
    
    
fig1,ax1 = plt.subplots(2,2,figsize=(12,6))


sns.histplot(x=age,kde=True,bins=10,ax=ax1[0,0])
ax1[0,0].set_title("Numpy without mean (CLT)")
ax1[0,0].set_xlabel("Retirement Age")


sns.histplot(x=mean,kde=True,bins=10,ax=ax1[0,1])
ax1[0,1].set_title("Numpy with mean (CLT)")
ax1[0,1].set_xlabel("Retirement Age")


sns.histplot(x=df["Retirement_Age"],kde=True,bins=10,ax=ax1[1,0])
ax1[1,0].set_title("Datasets without mean (CLT)")
ax1[1,0].set_xlabel("Retirement Age")


sns.histplot(x=mean1,kde=True,bins=10,ax=ax1[1,1])
ax1[1,1].set_title("Datasets with mean (CLT)")
ax1[1,1].set_xlabel("Retirement Age")


plt.tight_layout()
plt.show()