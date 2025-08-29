import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('Toyota.csv')

# plt.scatter(df["Age"],df["Price"],c="red")


# plt.title("Plotted")

# plt.xlabel("Age")

# plt.ylabel("Price")


# plt.hist(df["KM"],color="green",edgecolor="white",bins=20)
# plt.title("KiloMeter")

# plt.xlabel("KM")

# plt.ylabel("Frequency")
  
# Show plot

count = [979,120,12]

fuletype = ("Petrol","Diesel","CNG")

index = np.arange(len(fuletype))

bars =plt.bar(index,count,color=["red","blue","cyan"],label = fuletype)
plt.title("Bar plot")
plt.xlabel("Fule type")
plt.ylabel("Frequency")

plt.legend(bars,fuletype,title="fuletype")



plt.show()
