import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error

df = pd.read_csv("Revision/Data/linear_regression1.csv",index_col=0)
print(df)
print(df.describe())
print(df.info())
print(df.isna().sum())

df["Bedrooms"].fillna(df["Bedrooms"].mode()[0],inplace=True)
df["Age_years"].fillna(df["Age_years"].mean(),inplace=True)
df["Distance_km"].fillna(df["Distance_km"].mean(),inplace=True)
df["Price_Lakhs"].fillna(df["Price_Lakhs"].mean(),inplace=True)
print(df.isna().sum())

sns.scatterplot(df,x="Bedrooms",y="Price_Lakhs")
plt.show()
sns.scatterplot(df,x="Age_years",y="Price_Lakhs")
plt.show()
sns.scatterplot(df,x="Distance_km",y="Price_Lakhs")
plt.show()
sns.scatterplot(df,x="Area_sqft",y="Price_Lakhs")
plt.show()

sns.boxplot(data=df,x="Age_years")
plt.show()

# q1 = df["Age_years"].quantile(0.25)
# q3 = df["Age_years"].quantile(0.75)

# iqr = q3-q1

# low = q1-1.5*iqr
# hig = q3+1.5*iqr

# df["Age_years"] = np.where(df['Age_years']>hig,hig,np.where(df["Age_years"]<low,low,df["Age_years"]))

df["Age_years"] = winsorize(df["Age_years"],limits=[.1,.1])

sns.boxplot(data=df,x="Age_years")
plt.show()

sns.pairplot(df)
plt.show()

cor = df.corr()

sns.heatmap(cor,cmap="crest",annot=True)
plt.show()

x_features=["Area_sqft","Age_years","Distance_km"]
y_features=df["Price_Lakhs"]

x_train,x_test,y_train,y_test = train_test_split(df[x_features],y_features,test_size=.2,random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)  # auto-scaling happens here

r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)

print(r2)
print(mae)
print(mse)
print(rmse) 