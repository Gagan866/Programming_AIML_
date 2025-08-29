import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Gold_Price_USD': [1350, 1550, 1800, 1750, 1900],      # Avg price per ounce
    'Silver_Price_USD': [1300,1543,1600,1878,1890],    # Avg price per ounce
    
}

df = pd.DataFrame(data)
print(df)

# plt.scatter(df["Year"],df["Gold_Price_USD"],label="Gold")
# plt.scatter(df["Year"],df["Silver_Price_USD"],label="Silver")

# plt.xlim(2019,2023)
# plt.ylim(1350,1900)
# plt.xticks(range(2019,2025,1),rotation=45)
# plt.yticks(range(1350,2000,100))
# plt.legend(loc="best",bbox_to_anchor=(1,1))
# plt.grid(linestyle="--")
# plt.show()
plt.plot(df["Year"])
plt.show()