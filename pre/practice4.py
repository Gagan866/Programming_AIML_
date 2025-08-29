# import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5]
# y = [10, 20, 25, 30, 40]

# plt.figure(figsize=(6, 4))           # Whole figure
# plt.plot(x, y, label='Sample Data')  # The line
# plt.title("Anatomy of a Plot")       # Title
# plt.xlabel("X Axis Label")           # X-axis label
# plt.ylabel("Y Axis Label")           # Y-axis label
# plt.grid(True)                       # Grid
# plt.legend()                         # Legend
# plt.show()

import numpy as np
import time

# Generate two large arrays with 10 million elements
size = 10_000_000
a = np.random.rand(size)
b = np.random.rand(size)

# Addition
start = time.time()
add_result = a + b
print(add_result)
print("Addition time:", time.time() - start)

# Subtraction
start = time.time()
sub_result = a - b
print("Subtraction time:", time.time() - start)

# Multiplication
start = time.time()
mul_result = a * b
print("Multiplication time:", time.time() - start)

# Division
start = time.time()
div_result = a / b
print("Division time:", time.time() - start)
