import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay


df = pd.read_csv("DataSets/breast-cancer.csv")

print(df.shape)
print(df.info())
print(df.describe())
print(df.head(10))
print(df.columns)

print(df["diagnosis"].value_counts())

df.drop(columns=["id"],inplace=True) 

print(df.info())


df["diagnosis"] = df["diagnosis"].astype("category")


# features
# x = ['radius_mean', 'texture_mean',  'smoothness_mean', 'concavity_mean',
#         'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se',  'smoothness_se',
#        'concavity_se', 'symmetry_se',
#        'fractal_dimension_se', 'radius_worst', 'texture_worst',
#       'smoothness_worst',
#         'concavity_worst', 
#        'symmetry_worst', 'fractal_dimension_worst']
x = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

y = ['diagnosis']



sns.boxplot(df)
plt.show()


# plt.figure(figsize=(500,500))
# corr = df.corr(numeric_only=True)
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()

# df.drop(['perimeter_mean', 'area_mean','perimeter_worst', 'area_worst', 'perimeter_se', 'area_se', 'compactness_mean','concave points_mean', 'compactness_se', 'concave points_se','compactness_worst','concave points_worst'], axis=1, inplace=True)


def winsorize_data(data,limits=(0.01,0.01)):
    for col in data.select_dtypes(include=["float64"]):
         data.loc[:,col] = winsorize(data[col], limits=limits)
    return data


df_win = winsorize_data(df)


x1,y1 = train_test_split(df_win,random_state=42,test_size=.30,stratify=df['diagnosis'])


stdsclar = StandardScaler()

x1[x] = stdsclar.fit_transform(x1[x])
y1[x] = stdsclar.fit_transform(y1[x])

print(x1.head())
print(y1.head())


model = LogisticRegression(class_weight="balanced",max_iter=100,penalty="l2")

model.fit(x1[x],x1[y])

y_pred = model.predict(y1[x])


result = pd.DataFrame()
result["y_test"],result["y_pred"] = y1[y],y_pred
print(result)
print(model.predict_proba(y1[x]))

con = confusion_matrix(result['y_test'],result["y_pred"])

print(con)
print("_"*20)
print(classification_report(result['y_test'],result["y_pred"]))
print("_"*20)
print(accuracy_score(result['y_test'],result['y_pred']))



# sns.heatmap(con,cmap="crest",annot=True)
# plt.show()

RocCurveDisplay.from_estimator(model,)
