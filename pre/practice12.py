import numpy as np

# 0D - Scalar
s = np.array(5)

# 1D - Vector
v = np.array([1, 2, 3])

# 2D - Matrix
m = np.array([[1, 2], [3, 4]])

# 3D - Tensor
t = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("Tensor shape:", t.shape)  # Output: (2, 2, 2)
