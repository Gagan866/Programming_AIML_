import pandas as pd 

data = {
    "Region" : ["North","South"],
    "Product" : ["Phone" , "Laptop"],
    "Jan" : [23000,18000],
    "Feb" : [25000,21000],
    "Mar" : [27000,30000],
    "Apr" : [30000,25000]
}

df = pd.DataFrame(data)
print(df.reset_index().melt(id_vars=["Product"],value_vars=["Jan","Feb","Mar","Apr"]))

print(pd.pivot_table(df,index="Product",columns="Region",aggfunc="mean",fill_value=0,margins=True,margins_name="Total"))

 