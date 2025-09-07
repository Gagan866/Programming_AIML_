import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,confusion_matrix,accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


df = load_breast_cancer()

data = pd.DataFrame(df.data,columns=df.feature_names)
data["diagnosis"] =df.target
print(data.columns)

x_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
y = data["diagnosis"]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data[x_features])


# pca = PCA(n_components=20)
# x_pca = pca.fit_transform(x_scaled)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_)

pcas = []
for i in range(2,21):
       pca = PCA(n_components=i)
       x_pca = pca.fit_transform(x_scaled)
       e_var = pca.explained_variance_ratio_
       cum_sum = e_var.cumsum()
       pcas.append(cum_sum)
       print("c : ",cum_sum)
       i +=1

print(pcas)
print(len(pcas))
x_train,x_test,y_train,y_test = train_test_split(x_pca,y,random_state=42,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred))