import pandas as pd

# CSV
toyota_data_csv = pd.read_csv("Toyota.csv",index_col=0,na_values=["??","????"])
tc = toyota_data_csv.copy(deep=True)
print(toyota_data_csv.head(10))

# Excel
# toyota_data_excel = pd.read_excel("Toyota.xlsx",sheet_name="Sheet1",index_col=0,na_values=["??","????"])
# print(toyota_data_excel.head(10))

# Text
# toyota_data_txt = pd.read_table("Toyota.txt",index_col=0,na_values=["??","????"],delimiter="\t")
# print(toyota_data_txt.head(10)) 