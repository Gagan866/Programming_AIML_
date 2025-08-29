# import pandas as pd
# from sklearn.datasets import load_iris
# df = pd.read_csv("C:/183CS23010/Sem 5/AI&ML/Programming/data/bank.csv",sep=";")
# iris = load_iris()
# print(df[df["contact"].isna()])
# print(iris.info())

from sklearn.datasets import load_iris
import pandas as pd

# Load as Bunch (like a dictionary)
iris_bunch = load_iris()

# Convert to DataFrame
iris = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)

print(iris.head())
print(iris.info())
