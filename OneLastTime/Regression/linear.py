import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df1 = pd.read_csv("regression_data1.csv",index_col=0)

# print(df1)

df2 = pd.read_csv("regression_data2.csv",index_col=0)

# print(df2)

df_new = pd.concat([df1,df2],ignore_index=True)

print(df_new)
print(df_new.info())
print(df_new.describe())
print(df_new.isna().sum())
print(df_new.duplicated().sum())
print(df_new.columns)

df_new.drop_duplicates(inplace=True)
print(df_new.describe())

df_new[['City', 'Location', 'HouseType']] = df_new[['City', 'Location', 'HouseType']].astype("category")

sns.boxplot(df_new)
# plt.show()

q1 = df_new["Price"].quantile(.25)
q3 = df_new["Price"].quantile(.75)
iqr = q3-q1

l = q1-1.5*iqr
u = q3+1.5*iqr

outliers = df_new[(df_new["Price"]>u)|(df_new["Price"]<l)]
print(outliers)

df_new["Price"] = winsorize(df_new["Price"],limits=[.1,.1])

col = ['City', 'Location', 'HouseType', 'Area', 'Bedrooms', 'Bathrooms', 'Age',
       'Parking', 'Price']

sns.pairplot(df_new)
# plt.show()

sns.heatmap(df_new.corr(numeric_only=True),annot=True)
# plt.show()

df_new = df_new.drop(columns=["Bathrooms","Bedrooms"])

print(df_new)

print(pd.pivot_table(df_new,index="City",values="Price",aggfunc="mean"))

one = OneHotEncoder(drop="first",sparse_output=False)

encoded = one.fit_transform(df_new[['City', 'Location', 'HouseType']])

name = one.get_feature_names_out(['City', 'Location', 'HouseType'])

df_en = pd.DataFrame(encoded,columns=name,index=df_new.index)

print(df_en)


df = pd.concat([df_new,df_en],axis=1)
print(df)

df = df.drop(columns=['City', 'Location', 'HouseType'])
print(df)

num_col = ['Area','Age','Parking']

x = df.drop(columns="Price")
y = df["Price"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=42)

scl = StandardScaler()

xtrain[num_col] = scl.fit_transform(xtrain[num_col])
xtest[num_col] = scl.transform(xtest[num_col])

model = LinearRegression()

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

print(r2_score(ytest,ypred))

print(model.coef_)
print(model.intercept_)