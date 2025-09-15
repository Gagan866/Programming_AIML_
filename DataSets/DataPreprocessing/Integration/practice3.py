import pandas as pd

df1  = pd.read_csv("DataSets/Concat/file1.csv")
df2  = pd.read_csv("DataSets/Concat/file2.csv")

com = pd.concat([df1,df2],axis=0,ignore_index=True)

print(com)