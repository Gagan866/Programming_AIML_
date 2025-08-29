import pandas as pd

# df = pd.read_csv("DataSets/titanic.csv")

# print(df.info())
# print(df.describe())

# df = pd.DataFrame({
#     'Date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
#     'City': ['Bangalore', 'Mumbai', 'Bangalore', 'Mumbai'],
#     'Sales': [200, 150, 300, 180]
# # })
# df = {
#     'Department': ['IT', 'IT', 'HR', 'Finance', 'HR', 'Finance', 'IT'],
#     'Gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M'],
#     'Salary': [60000, 65000, 52000, 72000, 50000, 71000, 61000],
#     'Experience': [4, 3, 2, 7, 1, 6, 5]
# }
# data = pd.DataFrame(df)
# # pivot_df = df.pivot(index='Date', columns='City', values='Sales')
# # print(pivot_df)

# p = pd.pivot_table(data,index="Department",values=["Salary","Experience"],aggfunc={"Salary":"mean","Experience":"count"})
# print(p)
import pandas as pd

df = {
    'RegNo': [
        'REG001', 'REG002', 'REG003', 'REG004', 'REG005',
        'REG006', 'REG007', 'REG008', 'REG009', 'REG010',
        'REG011', 'REG012', 'REG013', 'REG014', 'REG015',
        'REG016', 'REG017', 'REG018', 'REG019', 'REG020'
    ],
    'Department': [
        'CS', 'EC', 'CE', 'ME', 'CS',
        'EC', 'CE', 'ME', 'CS', 'EC',
        'CE', 'ME', 'CS', 'EC', 'CE',
        'ME', 'CS', 'EC', 'CE', 'ME'
    ],
    'PMS': [
        78, 65, 89, 72, 81,
        90, 60, 74, 85, 68,
        79, 77, 62, 83, 80,
        88, 92, 66, 70, 75
    ],
    'Maths': [
        88, 72, 95, 70, 75,
        84, 65, 78, 80, 69,
        76, 85, 61, 82, 79,
        91, 90, 64, 73, 74
    ],
    'Gender': [
        'M', 'F', 'M', 'M', 'F',
        'F', 'M', 'F', 'M', 'M',
        'F', 'F', 'M', 'F', 'M',
        'M', 'F', 'M', 'F', 'M'
    ]
}

# Create DataFrame
data = pd.DataFrame(df)

# Show result
print(data)

pivot = pd.pivot_table(data,index="Department",columns="Gender",values=["PMS","Maths"],aggfunc={"PMS":"sum","Maths":"sum"})
print("_____________________________________________")
print(pivot)

pivot1 = pd.pivot_table(data,index="Department",columns="Gender",values=["PMS","Maths"],aggfunc={"PMS":"max","Maths":"max"})
print("_____________________________________________")
print(pivot1)



