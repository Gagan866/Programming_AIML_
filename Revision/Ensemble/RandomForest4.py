import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

df = pd.read_csv("Revision/Data/classification_practice_dataset.csv")

print(df)

print(df.describe())
print(df.info())
print(df.isna().sum())
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

df[["Income","CreditScore"]] = df[["Income","CreditScore"]].fillna(df[["Income","CreditScore"]].mean())

df["City"] = df["City"].fillna(df["City"].mode()[0])

print(df.isna().sum())

sns.boxplot(df)
# plt.show()

col_win = ["Age","Income","PastPurchases","HoursWorked"]

for i in col_win:
    df[i] = winsorize(df[i],limits=[.1,.1])

sns.boxplot(df)
# plt.show()

print(df)

df = pd.get_dummies(df,columns=["Department","City"],drop_first=True)

print(df.columns)

['Age', 'Income', 'Experience', 'HoursWorked', 'CreditScore',
       'PastPurchases', 'Purchased', 'Department_Fashion',
       'Department_Grocery', 'Department_Home', 'Department_Sports',
       'City_Chennai', 'City_Delhi', 'City_Hyderabad', 'City_Mumbai']

num_col = ['Age', 'Income', 'Experience', 'HoursWorked', 'CreditScore',
       'PastPurchases']


x = df.drop(columns="Purchased")
y=df["Purchased"]


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=.2,stratify=y)

sc = StandardScaler()

x_train[num_col] = sc.fit_transform(x_train[num_col])
x_test[num_col] = sc.transform(x_test[num_col])

model = RandomForestClassifier(max_depth=10,n_estimators=25,criterion="gini")

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


param = {
    "max_depth":[4,5,7,10],
    "n_estimators":[25,100,200],
    "criterion":["gini"]
}

grid = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param,scoring="accuracy",cv=5)

grid.fit(x_train,y_train)

best = grid.best_estimator_

y_pred1 = best.predict(x_test)

print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))