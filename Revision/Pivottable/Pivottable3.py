
import pandas as pd

# Sample data for Depot 1 (D1)
data1 = {
    'Vehicle ID': ['V1', 'V2', 'V3', 'V1', 'V2', 'V3', 'V1'],
    'Day': ['Mon', 'Mon', 'Mon', 'Tue', 'Tue', 'Tue', 'Wed'],
    'Fuel Usage (liters)': [50, 45, 60, 52, 48, 58, 51]
}
D1 = pd.DataFrame(data1)

# Sample data for Depot 2 (D2) - Ensure all vehicles have full week data in total
data2 = {
    'Vehicle ID': ['V2', 'V3', 'V1', 'V2', 'V3', 'V1', 'V2'],
    'Day': ['Wed', 'Wed', 'Thu', 'Thu', 'Thu', 'Fri', 'Fri'],
    'Fuel Usage (liters)': [46, 59, 53, 49, 61, 54, 47]
}
D2 = pd.DataFrame(data2)

# Sample data for Depot 3 (D3)
data3 = {
    'Vehicle ID': ['V3', 'V1', 'V2', 'V3', 'V1', 'V2', 'V3'],
    'Day': ['Fri', 'Sat', 'Sat', 'Sat', 'Sun', 'Sun', 'Sun'],
    'Fuel Usage (liters)': [57, 55, 50, 56, 54, 51, 55]
}
D3 = pd.DataFrame(data3)

# Display the dataframes (optional)
print("Depot 1 Data (D1):")
print(D1)
print("\nDepot 2 Data (D2):")
print(D2)
print("\nDepot 3 Data (D3):")
print(D3)

combine = pd.concat([D1,D2,D3],ignore_index=True)

print(combine)

print(pd.pivot_table(combine,index="Day",columns="Vehicle ID",values='Fuel Usage (liters)',aggfunc='mean'))

print(pd.pivot_table(combine,index="Vehicle ID",values='Fuel Usage (liters)',aggfunc="std"))