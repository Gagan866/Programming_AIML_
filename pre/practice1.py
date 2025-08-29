import pandas as pd
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}
mydataset1 = [1,2,3,4]

mydataset2 = {"x":10,"y":20,"z":30}

mydataset3 = [1,2,3,4,5,6,7,8]  
mydataset4 = [10,20,30,40,50,60,70,80]

myvar = pd.DataFrame(mydataset)
myvar1 = pd.Series(mydataset1,index=["a","b","c","d"])
myvar2 = pd.Series(mydataset2,index=["x",'z'])
myvar3 = pd.Series(mydataset3,mydataset4)

print(myvar)
print(myvar1)
print(pd.__version__)
print(myvar1)
print(myvar1["c"])
print(myvar2)
print(myvar3)