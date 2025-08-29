import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('Toyota.csv')

# Step 2: Basic Cleaning (if needed)
# Convert 'Doors' to string in case it's not parsed right
df['Doors'] = df['Doors'].astype(str)

# Step 3: Plot Age vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Price'], alpha=0.7, color='royalblue', edgecolor='k')

# Labels and title
plt.title('Car Price vs Age', fontsize=16)
plt.xlabel('Age (months)', fontsize=12)
plt.ylabel('Price (Euros)', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
