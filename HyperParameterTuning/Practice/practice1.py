import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("DataSets/forbes.csv")

print(df.info())
print(df.describe())
print(df.duplicated().value_counts())
print(df.isna().sum())
print(df.head(10))

print(df["Company"].value_counts())
print(df["Country"].value_counts())
print(df["Sector"].value_counts())
print(df["Industry"].value_counts())

