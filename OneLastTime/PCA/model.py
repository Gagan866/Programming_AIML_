import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
df = pd.read_csv("pca_data.csv", index_col=0)

# Encode categoricals
df[["Gender","EducationLevel","Department"]] = df[["Gender","EducationLevel","Department"]].astype("category")

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# One-hot encode
df = pd.get_dummies(df, columns=["EducationLevel", "Department"], drop_first=True)

print(df.head())

# -----------------------------------------------------
# Feature Split
# -----------------------------------------------------
x = df.drop(columns="PerformanceScore")
y = df["PerformanceScore"]

# -----------------------------------------------------
# Train-Test Split
# -----------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# SCALE ALL FEATURES (crucial for PCA)
# -----------------------------------------------------
scl = StandardScaler()
x_train_scaled = scl.fit_transform(x_train)
x_test_scaled = scl.transform(x_test)

# -----------------------------------------------------
# PCA Loop
# -----------------------------------------------------
range_ = range(2, 6)
r2_scores = []
explained = []

for i in range_:
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    model = LinearRegression()
    model.fit(x_train_pca, y_train)

    y_pred = model.predict(x_test_pca)

    r2_scores.append(r2_score(y_test, y_pred))
    explained.append(sum(pca.explained_variance_ratio_))

# -----------------------------------------------------
# Plot R2 and Explained Variance
# -----------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(range_, r2_scores, marker='o')
plt.title("R2 Score vs PCA Components")
plt.xlabel("n_components")
plt.ylabel("R2 Score")
plt.grid()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(range_, explained, marker='o', color='red')
plt.title("Explained Variance vs PCA Components")
plt.xlabel("n_components")
plt.ylabel("Variance Retained")
plt.grid()
plt.show()
