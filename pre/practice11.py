import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("Toyota.csv")

sns.set(style="darkgrid") 

plt.title("Price vs Age by Fuel Type", fontsize=14)
plt.xlabel("Age (Months)")
plt.ylabel("Price (Euros)")
plt.tight_layout()

# Use lmplot instead of regplot
sns.lmplot(x="Age", y="Price", hue="FuelType", data=df, palette="Set1", height=6, aspect=1.5)

sns.displot(df['Age'],bins=5)

sns.countplot(x=df["FuelType"],data=df)

plt.show()





plt.plot
plt.hist
plt.bar
plt.scatter

sns.regplot(Regression Plot)
sns.displot(Histogram)
sns.countplot(Bar)