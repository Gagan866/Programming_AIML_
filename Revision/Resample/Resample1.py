import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.date_range(start="1/1/2010",end="12/31/2024",freq="D")

print(data)

value = np.random.randint(10,100,size=len(data))

df = pd.DataFrame({"date":data,"value":value}).set_index("date")

print(df)

print(df.resample("M").sum())
print(df.resample("Y").sum())
print(df[df.index.month==6].resample("YE").sum())

june_trend = df[df.index.month == 6].resample("Y").sum()
june_trend.index = june_trend.index.year
print(june_trend)


years_data = df.resample("Y").sum()
years_data.index = years_data.index.year
years_data = years_data.reset_index(names="year")

print(years_data)

sns.barplot(data=years_data,x="year",y="value")
plt.show()

# Resample to monthly sum
monthly_data = df.resample("M").sum()
monthly_data["month"] = monthly_data.index.month
monthly_data["year"] = monthly_data.index.year


# Create 4x3 subplot grid
fig, axes = plt.subplots(4, 3, figsize=(16, 10))
axes = axes.flatten()

# Plot each month's data across years
for month in range(1, 13):
    month_data = monthly_data[monthly_data["month"] == month]
    axes[month - 1].bar(month_data["year"], month_data["value"])
    axes[month - 1].set_title(pd.to_datetime(str(month), format='%m').strftime('%B'))
    axes[month - 1].set_xlabel("Year")
    axes[month - 1].set_ylabel("Value")
    axes[month - 1].grid(True, linestyle='--', alpha=0.5)

# Adjust layout for better visibility
plt.tight_layout()
plt.show()