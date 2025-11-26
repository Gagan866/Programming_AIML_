import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Revision/Data/regression_practice_dataset.csv")

print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
print(df.duplicated().sum())

df[["Age","Experience"]]=df[["Age","Experience"]].fillna(df[["Age","Experience"]].mean())

df["Department"]=df["Department"].fillna(df["Department"].mode()[0])

print(df.isna().sum())

sns.boxplot(df)
plt.show()

q1 = df["Salary"].quantile(.25)
q3 = df["Salary"].quantile(.75)
iqr = q3-q1

l = q1-1.5*iqr
u = q3+1.5*iqr

outliers = df[(df["Salary"]>u) | (df["Salary"]<l)]

print(outliers)

df["Salary"] = winsorize(df["Salary"],limits=[.1,.1])

sns.boxplot(df)
# plt.show()

num_col = ["Age","Experience"]

cat_col = ["Department","Education"]

# LE = LabelEncoder()

# df["Department"] = LE.fit_transform(df["Department"])

# print(df)


# ohe = OneHotEncoder(drop="first",sparse_output=False)

# encoded = ohe.fit_transform(df[["Education","Department"]])

# # print(encoded)

# en_df = pd.DataFrame(encoded,columns=ohe.get_feature_names_out(["Education","Department"]),index=df.index)

# print(en_df)

# df = pd.concat([df,en_df],axis=1)
# print(df)

df = pd.get_dummies(df,columns=["Department","Education"],drop_first=True)
print(df)

df.drop(columns=["Education","Department"],inplace=True)
print(df)

x = df.drop(columns="Salary")

y = df["Salary"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)

SS = StandardScaler()
x_train[num_col] = SS.fit_transform(x_train[num_col])

x_test[num_col] = SS.transform(x_test[num_col])

print(x_train)
print(x_test)


model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(r2_score(y_test,y_pred))