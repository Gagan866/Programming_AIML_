import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.mstats import winsorize

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = pd.read_csv("classification_data.csv", index_col=0)

print(df.info())
print(df.isna().sum())

# -----------------------------------------------------------
# DATA CLEANING
# -----------------------------------------------------------

# Fill missing categorical values
df['InternetService'] = df['InternetService'].fillna(df['InternetService'].mode()[0])

# Convert to category dtype
df[['Gender','Partner','Dependents','Contract','InternetService','PaymentMethod']] = \
df[['Gender','Partner','Dependents','Contract','InternetService','PaymentMethod']].astype('category')

# Heatmap before encoding
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# Correct boxplot
sns.boxplot(data=df)
plt.show()

# Winsorize (safe limits: 2%)
df["TotalCharges"] = winsorize(df['TotalCharges'], limits=[0.02, 0.02])

# -----------------------------------------------------------
# ENCODING
# -----------------------------------------------------------

# Label encode binary columns
label_cols = ['Gender','Partner','Dependents']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-hot encode multi-class categorical features
df = pd.get_dummies(df, columns=['Contract','InternetService','PaymentMethod'], drop_first=True)

print(df.head())

# -----------------------------------------------------------
# FEATURE SELECTION
# -----------------------------------------------------------
num_features = ["MonthlyCharges", "TotalCharges", "Tenure"]

x = df.drop(columns="Churn")
y = df["Churn"]

# -----------------------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------------------
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# SCALING
# -----------------------------------------------------------
scl = StandardScaler()
xtrain.loc[:, num_features] = scl.fit_transform(xtrain[num_features])
xtest.loc[:, num_features] = scl.transform(xtest[num_features])

# -----------------------------------------------------------
# MODEL 1 — LOGISTIC REGRESSION
# -----------------------------------------------------------
log_model = LogisticRegression(max_iter=500)
log_model.fit(xtrain, ytrain)

log_pred = log_model.predict(xtest)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(ytest, log_pred))
print(classification_report(ytest, log_pred))

# -----------------------------------------------------------
# MODEL 2 — RANDOM FOREST
# -----------------------------------------------------------
rf = RandomForestClassifier(random_state=42)

rf.fit(xtrain, ytrain)
rf_pred = rf.predict(xtest)

print("\n=== Random Forest Results (Before Tuning) ===")
print("Accuracy:", accuracy_score(ytest, rf_pred))
print(classification_report(ytest, rf_pred))

# -----------------------------------------------------------
# GRIDSEARCHCV FOR RANDOM FOREST
# -----------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(xtrain, ytrain)

best_rf = grid.best_estimator_

print("\n=== Best Parameters From GridSearch ===")
print(grid.best_params_)

# Evaluate tuned model
grid_pred = best_rf.predict(xtest)

print("\n=== Random Forest Results (After Tuning) ===")
print("Accuracy:", accuracy_score(ytest, grid_pred))
print(classification_report(ytest, grid_pred))
    