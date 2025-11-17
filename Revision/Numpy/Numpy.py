import numpy as np

sales = np.array([[1200,1500,1000,1800],
                  [800,950,1100,1200],
                  [1500,1600,1700,1750]])


print("Total Sales Each Region : ",sales.sum(1))
print("Average sales across all regions : ",sales.mean())
print("Highest sales month column wise : ",sales.max(0))
print("Std of sales value : ",sales.std())
print("Minimum sales value : ",sales.min())


total_sales = np.sum(sales,axis=1)
print(total_sales)
avg_sales = np.mean(sales)
print(avg_sales)
hig_sales = np.max(sales,axis=0)
print(hig_sales)
std_sales = np.std(sales)
print(std_sales)
min_sales = np.min(sales)
print(min_sales)

