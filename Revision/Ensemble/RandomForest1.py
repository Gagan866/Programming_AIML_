import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
df = pd.read_csv("Revision/Data/Iris.csv", index_col=0)

print(df.head())
print(df.describe())
print(df.info())
print(df.isna().sum())

# ---------------------------------------------------
# 2. Outlier Visualization
# ---------------------------------------------------
sns.boxplot(df)
plt.title("Boxplot of Iris Features")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------
# 3. Encode Target
# ---------------------------------------------------
df["Species"] = df["Species"].astype("category")
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

# ---------------------------------------------------
# 4. Winsorization
# ---------------------------------------------------
df["SepalWidthCm"] = winsorize(df["SepalWidthCm"], limits=[0.1, 0.1])

# ---------------------------------------------------
# 5. Feature Selection
# ---------------------------------------------------
x_features = ["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    df[x_features], y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------
# 6. ⭐ Random Forest Model
# ---------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    criterion="gini",        # splitting criterion
    max_depth=None,          # fully grown trees
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# 7. Predictions & Evaluation
# ---------------------------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# 8. Feature Importance Plot
# ---------------------------------------------------
importances = model.feature_importances_
plt.barh(x_features, importances, color="green")
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.show()
