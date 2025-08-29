import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df = pd.read_csv("DataSets/literacy.csv")

print(df.head(10))


print("___"*10)
print("Rural")
print("___"*10)
print("Measures of Central Tendency")
print("___"*10)

r_male_avg = df["R_Male"].mean()
r_female_avg = df["R_Female"].mean()
r_male_med = df["R_Male"].median()
r_female_med = df["R_Female"].median()
r_male_mod = df["R_Male"].mode()
r_female_mod = df["R_Female"].mode()
print("Male mean",r_male_avg)
print("Female mean",r_female_avg)
print("Male median",r_male_med)
print("Female median",r_female_med)
print("Male Mode",r_male_mod)
print("Female Mode",r_female_mod)

print("___"*10)
print("Measures of Dispersion")
print("___"*10)

r_male_min = df["R_Male"].min()
r_female_min = df["R_Female"].min()
r_male_max = df["R_Male"].max()
r_female_max = df["R_Female"].max()
r_male_range = r_male_max - r_male_min
r_female_range = r_female_max - r_female_min
q1_r_male = np.quantile(df["R_Male"],.25)
q3_r_male = np.quantile(df["R_Male"],.75)
q1_r_female = np.quantile(df["R_Female"],.25)
q3_r_female = np.quantile(df["R_Female"],.75)
iqr_r_male = q3_r_male-q1_r_male
iqr_r_female = q3_r_female-q1_r_female
std_r_male = np.std(df["R_Male"])
std_r_female = np.std(df["R_Female"])
var_r_male = np.var(df["R_Male"])
var_r_female = np.var(df["R_Female"])
print("Male Range",r_male_range)
print("Female Range",r_female_range)
print("Male IQR",iqr_r_male)
print("Female IQR",iqr_r_female)
print("Male Std",std_r_male)
print("Female Std",std_r_female)
print("Male Var",var_r_male)
print("Female Var",var_r_female)


print("___"*10)

print("Urban")
print("___"*10)
print("Measures of Central Tendency")
print("___"*10)
u_male_avg = df["U_Male"].mean()
u_female_avg = df["U_Female"].mean()
u_male_med = df["U_Male"].median()
u_female_med = df["U_Female"].median()
u_male_mod = df["U_Male"].mode()
u_female_mod = df["U_Female"].mode()
print("Male mean",u_male_avg)
print("Female mean",u_female_avg)
print("Male median",u_male_med)
print("Female median",u_female_med)
print("Male Mode",u_male_mod)
print("Female Mode",u_female_mod)

print("___"*10)
print("Measures of Dispersion")
print("___"*10)

u_male_min = df["U_Male"].min()
u_female_min = df["U_Female"].min()
u_male_max = df["U_Male"].max()
u_female_max = df["U_Female"].max()
u_male_range = u_male_max - u_male_min
u_female_range = u_female_max - u_female_min
q1_u_male = np.quantile(df["U_Male"],.25)
q3_u_male = np.quantile(df["U_Male"],.75)
q1_u_female = np.quantile(df["U_Female"],.25)
q3_u_female = np.quantile(df["U_Female"],.75)
iqr_u_male = q3_u_male-q1_u_male
iqr_u_female = q3_u_female-q1_u_female
std_u_male = np.std(df["U_Male"])
std_u_female = np.std(df["U_Female"])
var_u_male = np.var(df["U_Male"])
var_u_female = np.var(df["U_Female"])
print("Male Range",u_male_range)
print("Female Range",u_female_range)
print("Male IQR",iqr_u_male)
print("Female IQR",iqr_u_female)
print("Male Std",std_u_male)
print("Female Std",std_u_female)
print("Male Var",var_u_male)
print("Female Var",var_u_female)


fig,ax = plt.subplots(2,2,figsize=(16,8))

sns.barplot(data=df,x="State",y="R_Male",ax=ax[0,0],palette="deep")
ax[0, 0].set_title("Rural Male Literacy Distribution")
sns.barplot(data=df,x="State",y="R_Female",ax=ax[0,1],palette="deep")
ax[0, 1].set_title("Rural Female Literacy Distribution")
sns.barplot(data=df,x="State",y="U_Male",ax=ax[1,0],palette="deep")
ax[1, 0].set_title("Urban Male Literacy Distribution")
sns.barplot(data=df,x="State",y="U_Female",ax=ax[1,1],palette="deep")
ax[1, 1].set_title("Urban Female Literacy Distribution")

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,5))
sns.lineplot(data=df,x="State",y="R_Male",label="R_Male")
sns.lineplot(data=df,x="State",y="R_Female",label="R_Female")
sns.lineplot(data=df,x="State",y="U_Male",label="U_Male")
sns.lineplot(data=df,x="State",y="U_Female",label="U_Female")

plt.tight_layout()
plt.show()