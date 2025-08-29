# # Supervised: Predict salary from years of experience
# from sklearn.linear_model import LinearRegression

# X = [[1], [2], [3], [4]]  # Years
# y = [30000, 40000, 50000, 60000]  # Salary

# model = LinearRegression()
# model.fit(X, y)
# print(model.predict([[5]]))  # Predict for 5 years


# Unsupervised: Cluster students by marks
# from sklearn.cluster import KMeans
# import numpy as np

# X = np.array([[50], [55], [60], [90], [92], [95]])  # Marks
# model = KMeans(n_clusters=2)
# model.fit(X)
# print(model.labels_)  # 0 or 1 – two groups

# import random

# # Environment setup
# states = ["A", "B", "C", "D", "E"]
# actions = ["left", "right"]
# rewards = {"A": 0, "B": 0, "C": 0, "D": 1, "E": -1}

# # Q-table initialization
# Q = {}
# for state in states:
#     Q[state] = {action: 0 for action in actions}

# # Training (Q-learning)
# alpha = 0.1   # Learning rate
# gamma = 0.9   # Discount factor

# for episode in range(100):
#     state = "B"
#     while state != "D" and state != "E":
#         action = random.choice(actions)
#         next_state = state

#         if action == "right" and states.index(state) < len(states)-1:
#             next_state = states[states.index(state)+1]
#         elif action == "left" and states.index(state) > 0:
#             next_state = states[states.index(state)-1]

#         reward = rewards[next_state]
#         best_future_q = max(Q[next_state].values())

#         # Q-learning formula
#         Q[state][action] += alpha * (reward + gamma * best_future_q - Q[state][action])
#         state = next_state

# # Final learned Q-values
# from pprint import pprint
# pprint(Q)

from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score
import numpy as np

# Load data (Digits dataset)
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Make some labels -90% unlabeled
rng = np.random.RandomState(42)
n_total = len(y)
n_labeled = int(0.1 * n_total)

# Create unlabeled array
y_semi = np.copy(y)
unlabeled_indices = rng.choice(n_total, size=(n_total - n_labeled), replace=False)
y_semi[unlabeled_indices] = -1  # -1 means "unlabeled" in scikit-learn

# Train a Label Spreading model
model = LabelSpreading(kernel='knn', n_neighbors=5)
model.fit(X, y_semi)

# Evaluate on all labels
predictions = model.transduction_
accuracy = accuracy_score(y, predictions)
print("Semi-Supervised Accuracy:", round(accuracy * 100, 2), "%")
