import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from scipy.stats.mstats import winsorize

df = pd.read_csv("Revision/Data/Iris.csv",index_col=0)
print(df)
print(df.describe())
print(df.info())
print(df.isna().sum())

sns.boxplot(df)
plt.show()

# sns.pairplot(df)
# plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

df["Species"] = df["Species"].astype("category")

en = LabelEncoder()

df["Species"] = en.fit_transform(df["Species"])

df["SepalWidthCm"] = winsorize(df["SepalWidthCm"],limits=[.1,.1])

x_features = ["SepalLengthCm","SepalWidthCm","PetalWidthCm"]
y_features = df["Species"]

x_train,x_test,y_train,y_test = train_test_split(df[x_features],y_features,random_state=42,test_size=.2,stratify=y_features)

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
# ------------------------------------------------------------
# ⭐ GridSearchCV for Logistic Regression
# ------------------------------------------------------------

param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 5, 10],
    "solver": ["liblinear", "saga"],
}

# Create model
log_reg = LogisticRegression(max_iter=500)

# Grid search
grid = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=5,              # 5-fold cross validation
    scoring="accuracy",
    n_jobs=-1,         # use all cores
    verbose=1
)

# Fit grid search
grid.fit(x_train_scaled, y_train)

# Best model
print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Predict using best model
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test_scaled)

# Evaluation
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
