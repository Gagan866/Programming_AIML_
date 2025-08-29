import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {"months" : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
"product_a" : [100, 110, 130, 125, 140, 150],
"product_b" : [120, 115, 125, 110, 130, 135],
"product_c" : [90, 95, 105, 100, 120, 125]}

df = pd.DataFrame(data)

print(df.head())

fig,ax = plt.subplots(3,1,figsize=(10,4))
plt.title("Frequency")

ax[0].plot(df["months"],df["product_a"],label="Product 1",marker="*")
ax[0].set_title("Product 1")

ax[1].plot(df["months"],df["product_b"],label="Product 2",marker="*")
ax[1].set_title("Product 2")


ax[2].plot(df["months"],df["product_c"],label="Product 3",marker="*")
ax[2].set_title("Product 3")

plt.tight_layout()

plt.show()

# plt.plot(df["months"],df["product_a"],label="Product 1",marker="*")
# plt.plot(df["months"],df["product_b"],label="Product 2",marker="*")
# plt.plot(df["months"],df["product_c"],label="Product 3",marker="*")
# # plt.annotate()
# plt.legend()
# plt.annotate(
#     "Peak: 150",                             # Text
#     xy=("Jun", 150),                         # Point to annotate
#     xytext=("May", 160),                     # Position of the text
#     arrowprops=dict(arrowstyle="->")         # Arrow style
# )
# plt.show()