import pandas as pd

data = {
    "Month" : ["Jan","Jan","Feb","Feb"],
    "Product" : ["A","B","A","B"],
    "Sales" : [100,200,150,250],
    "Profit" : [10,70,50,100]
}

df = pd.DataFrame(data)

print(df)

pivot1 = (pd.pivot_table(df,index="Month",columns="Product",values="Sales",aggfunc="sum"))
print(pivot1)
pivot2= (pd.pivot_table(df,index="Product",columns="Month",values="Sales",aggfunc="sum"))
print(pivot2)

pivot3 = (pd.pivot_table(df,index="Month",columns="Product",values=["Sales","Profit"],aggfunc="sum",sort=False))
print(pivot3)

p1 = pivot1.reset_index().melt(id_vars=["Month"],var_name="Product",value_name="Sales")
print(p1)
p2 = pivot2.reset_index().melt(id_vars=["Product"],var_name="Month",value_name="Sales")
print(p2)
p3 = pivot3.reset_index().melt(id_vars=["Month"],var_name=["Sales","Profit","Product"])
print(p3)