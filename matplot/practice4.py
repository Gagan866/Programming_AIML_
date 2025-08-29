import matplotlib.pyplot as plt
import pandas as pd







df = pd.read_csv("DataSets/Titanic.csv",na_values=["??","????"])
drop=df.dropna()
print(df.info())
print(drop.head(10))


pclass_data = df["Pclass"]

# Count passengers in each class manually
pclass_counts = {1: 0, 2: 0, 3: 0}
for pclass in pclass_data:
    pclass_counts[pclass] += 1
    

# X and Y for the bar plot
x_labels = ["1st Class", "2nd Class", "3rd Class"]
x = list(pclass_counts.keys())  # [1, 2, 3]
y = list(pclass_counts.values())  # [count of 1st, 2nd, 3rd]


# Create the figure
plt.figure(figsize=(6, 4))

# Create the bar chart
plt.bar(x_labels, y, color='gold')

# Add labels and title
plt.title("Number of Passengers by Class (Titanic)", fontsize=14)
plt.xlabel("Passenger Class", fontsize=12)
plt.ylabel("Number of Passengers", fontsize=12)

# Add grid and adjust layout
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Show the plot
plt.show()
