import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# y = f(X) = x^2


# data = pd.DataFrame({
#     "X_INPUT": [
#         1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#         21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
#         31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#         41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#         51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
#         61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
#         71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
#         81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
#         91, 92, 93, 94, 95, 96, 97, 98, 99, 100
#     ],
#     "TARGET": [
#         3, 6, 11, 18, 27, 38, 51, 66, 83, 102,
#         121, 142, 165, 190, 217, 246, 277, 310, 345, 382,
#         421, 462, 505, 550, 597, 646, 697, 750, 805, 862,
#         921, 982, 1045, 1110, 1177, 1246, 1317, 1390, 1465, 1542,
#         1621, 1702, 1785, 1870, 1957, 2046, 2137, 2230, 2325, 2422,
#         2521, 2622, 2725, 2830, 2937, 3046, 3157, 3270, 3385, 3502,
#         3621, 3742, 3865, 3990, 4117, 4246, 4377, 4510, 4645, 4782,
#         4921, 5062, 5205, 5350, 5497, 5646, 5797, 5950, 6105, 6262,
#         6421, 6582, 6745, 6910, 7077, 7246, 7417, 7590, 7765, 7942,
#         8121, 8302, 8485, 8670, 8857, 9046, 9237, 9430, 9625, 9822
#     ]
# })

data = pd.read_csv("DataSets/Models/Sqrtnum.csv")
print(data.head())

x = data[["X_INPUT"]]
y = data["TARGET"]


x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)
print(x_train)
print("-"*20)
print(x_test)
print("-"*20)
print(y_test)
print("-"*20)
print(y_train)
print("-"*20)


model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

result = pd.DataFrame()

# result["x_test","y_test","y_pred"], = x_test,y_test,y_pred

result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred


print(result.head(10))

    