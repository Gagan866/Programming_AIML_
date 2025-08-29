import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.svm import SVR

# Loading Data
df = pd.read_csv("DataSets/L_Reg/agriculture_dataset.csv")

# Preview Data
print(df.head(10))

# Understand the data structure
print(df.info())

# Understanding the data statistical summary
print(df.describe())

feature_set = df.columns
print(feature_set)