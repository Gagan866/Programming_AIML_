# import numpy as np


# array = np.array([[[1,2,3,4]],[[5,6,7,8]],[[1,2,3,4]]])

# arr1 = np.arange(0,20,2)
# arr2 = np.linspace(0,10,20)
# arr3 = np.zeros((2,3))
# arr4 = np.random.randint(0,100,(3,3))
# print(array.ndim)
# print(arr2)
# print(arr3)
# print(np.sum(array))
# print(np.unique(array))
# print(arr4)
# print(np.min(arr4))
# print(np.max(arr4))
# print(array)
# print(np.sum(array,axis=2))

# array = np.array([1,2,3,4])
# print(np.power(array,2))
# print(np.power(array,3))
# print(np.subtract(array,10))



import time
import random
import numpy as np

ele=np.random.rand(100)
print(ele)
arr = np.random.randint(0,100,10000000)
print(arr)
start_time = time.time()
vec = np.power(arr,2)

end_time = time.time()    
print(f"Computation time :{end_time-start_time:.4f}")

start_time1 = time.time()

arr1=[]
for i in arr:
    arr1.append(i**2)
end_time2 = time.time()    
print(f"Computation time :{end_time2-start_time1:.4f}")

# print(f"Start time :{start_time:.2f}")
# print(end_time)
