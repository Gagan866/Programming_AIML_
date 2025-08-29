import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Gold_Price_USD': [1350, 1550, 1800, 1750, 1900],      # Avg price per ounce
    'Silver_Price_USD': [1300,1543,1600,1878,1890],    # Avg price per ounce
    
}

df = pd.DataFrame(data)
print(df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plotting on the first axis
axes[0].plot([1, 2, 3], [3, 2, 1])
axes[0].set_title("Plot 1")

# Plotting on the second axis
axes[1].plot([1, 2, 3], [1, 2, 3])
axes[1].set_title("Plot 2")

plt.tight_layout()
plt.show()


# import matplotlib.pyplot as plt

# # Create 2 subplots (1 row, 2 columns)
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# # Plotting on the first axis
# axes[0].plot([1, 2, 3], [3, 2, 1])
# axes[0].set_title("Plot 1")

# # Plotting on the second axis
# axes[1].plot([1, 2, 3], [1, 2, 3])
# axes[1].set_title("Plot 2")

# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))

# # First subplot (1 row, 2 columns, 1st plot)
# plt.subplot(1, 2, 1)
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.title("Left Plot")

# # Second subplot
# plt.subplot(1, 2, 2)
# plt.plot([1, 2, 3], [6, 5, 4])
# plt.title("Right Plot")

# plt.tight_layout()  # Avoid overlap
# plt.show()




# import numpy as np 
# branch = ['CSE','CE','EC','ME']
# student_count = [100,30,50,45]
# index = np.arange(len(branch))
# plt.xlabel('BRANCH')
# plt.ylabel('COUNT OF STUDENTS')
# plt.xticks(index,branch,rotation=60)
# plt.bar(index,student_count,color = ['blue','red','cyan','yellow'])
# # plt.colorbar(label='Color Scale')
# plt.show()