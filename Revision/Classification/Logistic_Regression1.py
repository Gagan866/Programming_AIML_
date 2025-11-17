import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from scipy.stats.mstats import winsorize

df = pd.read_csv("Revision/Data/Iris.csv",index_col=0)
print(df)
print(df.describe())
print(df.info())
print(df.isna().sum())

sns.boxplot(df)
plt.show()

# sns.pairplot(df)
# plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

df["Species"] = df["Species"].astype("category")

en = LabelEncoder()

df["Species"] = en.fit_transform(df["Species"])

df["SepalWidthCm"] = winsorize(df["SepalWidthCm"],limits=[.1,.1])

x_features = ["SepalLengthCm","SepalWidthCm","PetalWidthCm"]
y_features = df["Species"]

x_train,x_test,y_train,y_test = train_test_split(df[x_features],y_features,random_state=42,test_size=.2,stratify=y_features)

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

model = LogisticRegression(
    penalty="l2",      
    C=1.0,            
    solver="lbfgs",   
    max_iter=200)


model.fit(x_train_scaled,y_train)

y_pred = model.predict(x_test_scaled)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
