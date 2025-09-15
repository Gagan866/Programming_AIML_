# import pandas as pd

# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Example dataset: 20 students, mostly passing
# data = {
#     "marks": [98, 67, 75, 82, 91, 56, 62, 43, 88, 73, 51, 63, 84, 95, 78, 49, 36, 81, 60, 92],
#     "attendance": [90, 85, 92, 88, 91, 70, 72, 60, 89, 80, 66, 77, 83, 94, 78, 58, 63, 86, 75, 93],
#     "result": ["Pass", "Pass", "Pass", "Fail", "Pass", "Fail", "Pass", "Fail", "Pass", "Pass",
#                "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass"]
# }

# df = pd.DataFrame(data)
# print(df)

# X = df[["marks", "attendance"]]
# y = df["result"]

# # Split WITHOUT stratify (random, may imbalance!)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=1, shuffle=True, stratify=None
# )

# # Split WITH stratify (class proportions preserved)
# X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
#     X, y, test_size=0.3, random_state=1, shuffle=True, stratify=y
# )

# print("Train-set result counts WITHOUT stratify:\n", y_train.value_counts())
# print("\nTest-set result counts WITHOUT stratify:\n", y_test.value_counts())

# print("\nTrain-set result counts WITH stratify:\n", y_train_strat.value_counts())
# print("\nTest-set result counts WITH stratify:\n", y_test_strat.value_counts())

# # Visualize class balance
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# sns.countplot(x=y_train)
# plt.title('Train—No Stratify')
# plt.subplot(122)
# sns.countplot(x=y_train_strat)
# plt.title('Train—With Stratify')
# plt.tight_layout()
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create an intentionally imbalanced dataset (18 Pass, 2 Fail)
data = {
    "marks": [
        98, 86, 77, 92, 66, 88, 93, 75, 82, 91, 63, 79, 72, 81, 68, 85, 78, 69, 94, 73,
        51, 63, 84, 95, 78, 49, 36, 81, 60, 92, 55, 89, 70, 77, 83, 62, 80, 61, 76, 74,
        59, 87, 90, 69, 90, 82, 68, 91, 88, 66, 67, 54, 73, 56, 85, 72, 83, 60, 84, 58,
        57, 77, 93, 70, 65, 78, 80, 64, 98, 68, 90, 92, 79, 73, 66, 89, 61, 74, 71, 88,
        82, 75, 63, 87, 60, 85, 77, 93, 68, 88, 84, 80, 69, 78, 81, 72, 64, 91, 62, 79
    ],
    "attendance": [
        90, 85, 92, 88, 91, 70, 81, 92, 80, 89, 84, 77, 91, 93, 87, 90, 75, 81, 91, 85,
        66, 77, 83, 94, 78, 58, 63, 86, 75, 93, 88, 77, 91, 80, 89, 78, 66, 83, 90, 94,
        79, 76, 95, 81, 88, 90, 85, 93, 89, 73, 72, 80, 91, 62, 70, 76, 81, 68, 84, 87,
        75, 81, 93, 85, 82, 88, 77, 93, 79, 66, 90, 92, 71, 68, 91, 94, 76, 82, 90, 87,
        88, 75, 63, 86, 84, 91, 80, 89, 70, 74, 83, 95, 81, 78, 90, 93, 87, 80, 92, 88
    ],
    "result": [
        "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

X = df[["marks", "attendance"]]
y = df["result"]

# Step 2a: Split WITHOUT stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, shuffle=True, stratify=None
)

print("WITHOUT stratify:")
print("Train:", y_train.value_counts().to_dict())
print("Test:", y_test.value_counts().to_dict())

# Step 2b: Split WITH stratify
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.3, random_state=1, shuffle=True, stratify=y
)

print("\nWITH stratify:")
print("Train:", y_train_s.value_counts().to_dict())
print("Test:", y_test_s.value_counts().to_dict())

# Step 3: Visual comparison
plt.figure(figsize=(10,4))
plt.subplot(121)
sns.countplot(x=y_train, palette="cool")
plt.title("Train—No Stratify")
plt.subplot(122)
sns.countplot(x=y_train_s, palette="cool")
plt.title("Train—With Stratify")
plt.tight_layout()
plt.show()
