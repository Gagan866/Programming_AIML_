import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split

# iris = load_iris()

# x = pd.DataFrame(iris.data,columns=iris.feature_names)

# print(x)

# y = iris.target

# xt,yt,xt1,yt1 = train_test_split(x,y,test_size=0.2)

# print(xt.shape)
# print(yt.shape)
# print(xt1.shape)
# print(yt1.shape)

# df = pd.read_csv("DataSets/titanic.csv")

# x = df["Survived"]

# features = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

# y = df[features]

# xt,xt1,yt,yt1 = train_test_split(x,y,test_size=0.2)

# print(xt.shape)
# print(yt.shape)
# print(xt1.shape)
# print(yt1.shape)

df = pd.read_csv("DataSets/Boston.csv",index_col=0)

print(df.info())
print(df.head())
print(df.columns)

features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad','ptratio', 'black', 'lstat', 'medv']

x = df[features]

y = df["tax"]

xt,xt1,yt,yt1 = train_test_split(x,y,test_size=0.2)

print(xt.shape)
print(yt.shape)
print(xt1.shape)
print(yt1.shape)
