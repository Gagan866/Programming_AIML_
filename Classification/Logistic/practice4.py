import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset
df = pd.read_csv("DataSets/breast-cancer.csv")

# 2. Drop 'id' column (not useful for prediction)
df.drop(columns=["id"], inplace=True)

# 3. Encode target variable
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# 4. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 5. Select features and target
feature_columns = [col for col in df.columns if col != "diagnosis"]
X = df[feature_columns]
y = df["diagnosis"]

# 6. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # returns a numpy array

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Build and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Make predictions
y_pred = model.predict(X_test)

# 10. Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: Check model coefficients
coefficients = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_[0]
})
print("Model Coefficients:\n", coefficients)


