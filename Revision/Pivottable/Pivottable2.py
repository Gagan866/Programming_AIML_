import pandas as pd

data = {
    "Student" : ["Riya" , "Arjun" , "Neha"],
    "Subject" : ["Math","Science", "English"],
    "Marks" : [78,64,82],
    "Gender" : ["F","M","F"]
}

df = pd.DataFrame(data).set_index("Student")

print(df)

marks = [0,25,50,75,100]
bins = ["0-25","25-50","50-75","75-100"]

df["Category"] = pd.cut(df["Marks"],bins=marks,labels=bins)

print(df)

# print(pd.pivot_table(df,index="Gender",columns="Subject",values="Marks",aggfunc="count",fill_value=0,margins=True))

pivot = (pd.pivot_table(df,index="Category",columns=["Subject","Gender"],aggfunc="count",fill_value=0))

print(pivot)

print(df.reset_index().melt(id_vars="Student", value_vars=["Marks","Gender","Subject","Category"]))