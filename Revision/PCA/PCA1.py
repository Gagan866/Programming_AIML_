import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

df = pd.read_csv("Revision/Data/pca1.csv")
print(df)

x_features = ["MathScore","ReadingScore","WritingScore","Attendance","StudyHours"]

y_features = df['IQ']

X_train, X_test, y_train, y_test = train_test_split(
    df[x_features],
    y_features,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------
# Scaling
# ------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------
# PCA
# ------------------------------------------------------
for n in range(1, 6):   # Try PCA from 1 to 5 components
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    
    preds = model.predict(X_test_pca)
    score = r2_score(y_test, preds)

    print(f"PCA Components: {n}, R2 Score: {score:.4f}")