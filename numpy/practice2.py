import numpy as np
import functools as fun
import math
# arr = np.random.randint(100, 200, 50)

# lst = list(map(int, filter(lambda x: x > 150, arr)))
# lst1 = fun.reduce(lambda c,_ : c+1 , lst,0)
# print(lst1)

arr = np.random.randint(100, 500, 50)

sqrts = list(map(int,filter(lambda x:math.isqrt(x)**2 ==x,arr)))
print(sqrts)
count = fun.reduce(lambda c,_:c+1,sqrts,0)
print(count)
