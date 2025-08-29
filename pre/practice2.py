import pandas as pd

mydata = ["Apple","Mango","Banana"]
myprice = [100,200,300]
myquantity = [1,2,3]

sd = pd.Series(mydata)
sp = pd.Series(myprice)
sq = pd.Series(myquantity)

data={
    "Fruits":sd,"Price":sp,"Quantity":sq
   
    }

df = pd.DataFrame(data)

print(df)
