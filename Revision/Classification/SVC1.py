import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize

# Load dataset
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

# Check basics
print(df.head())
print(df.describe())
print(df.info())
print(df.isna().sum())

# Outlier visualization
sns.boxplot(df)
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
plt.show()

# Convert target to categorical
df["Species"] = df["Species"].astype("category")

# Label encode target
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# Winsorize Sepal Width
df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# Feature selection
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df[x_features], y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# ⭐ SVM Model (Support Vector Machine)
# --------------------------------------------

model = SVC(
    kernel="rbf",     # try 'linear', 'poly', or 'rbf'
    C=1.0,
    gamma="scale",    # auto scaling for RBF kernel
)

# Train
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
