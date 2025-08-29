import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

df = pd.read_csv("DataSets/Birthweight_reduced_kg_R.csv")

# sns.barplot(data=df,x="Gestation",y="Birthweight")

# sns.pairplot(data=df)
# plt.tight_layout
# plt.show()

print(st.shapiro(df["Birthweight"]))
print(st.chisquare(df["Birthweight"]))
