import pandas as pd

data = {
    "Gender" : ["M","F","M","F"],
    "Subject" : ["Math","Math","Science","Science"],
    "Marks" : [80,90,85,95]
}

df = pd.DataFrame(data)
print(df)
pivot = pd.pivot_table(df,index="Subject",columns="Gender",values="Marks",aggfunc="mean")
print(pivot)

print(pd.melt(pivot.reset_index(),id_vars="Subject",var_name="Gender",value_name="Marks"))