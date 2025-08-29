# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample DataFrame
df = pd.DataFrame({
    'department': np.random.choice(['CS', 'EC', 'Civil', 'Mechanical'], size=100),  # department: Randomly chosen from four categories
    'salary': np.random.randint(25000, 100000, size=100),  # salary: Random integers between 25000 and 100000
    'experience': np.random.randint(1, 20, size=100),  # experience: Random integers between 1 and 20 years
    'gender': np.random.choice(['Male', 'Female'], size=100),  # gender: Randomly chosen Male or Female
    'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Bangalore'], size=100)  # city: Randomly chosen from four cities
})

# 1️⃣ COUNT PLOT
sns.countplot(
    x='department',  # x-axis categories
    data=df,  # source DataFrame
    hue='gender',  # subgroup bars by gender
    palette='pastel',  # set color palette
    order=['CS', 'EC', 'Civil', 'Mechanical']  # define order of categories on x-axis
)
plt.title("Department Distribution by Gender")
plt.show()

# 2️⃣ BOX PLOT
sns.boxplot(
    x='department',  # x-axis categories
    y='salary',  # y-axis numerical variable
    hue='gender',  # split boxes within each department by gender
    data=df,  # source DataFrame
    palette='coolwarm',  # color palette
    linewidth=2,  # thickness of box borders
    width=0.5,  # width of each box
    fliersize=4,  # size of outlier points
    whis=1.5  # whisker length multiplier (1.5 * IQR)
)
plt.title("Salary Distribution Across Departments by Gender")
plt.show()

# 3️⃣ SCATTER PLOT
sns.scatterplot(
    x='salary',  # x-axis numerical variable
    y='experience',  # y-axis numerical variable
    hue='department',  # color points based on department
    style='gender',  # marker style based on gender
    size='experience',  # size of points based on experience
    sizes=(20,200),  # range of point sizes
    palette='deep',  # color palette
    data=df  # source DataFrame
)
plt.title("Scatter Plot of Salary vs Experience")
plt.show()

# 4️⃣ HISTOGRAM (Matplotlib)
plt.hist(
    df['salary'],  # data to plot
    bins=15,  # number of bins
    color='purple',  # bar color
    edgecolor='black',  # color of bar edges
    alpha=0.7  # transparency
)
plt.title("Salary Distribution Histogram")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# 5️⃣ HISTOGRAM (Seaborn)
sns.histplot(
    df['salary'],  # data to plot
    bins=20,  # number of bins
    color='green',  # bar color
    kde=True  # include Kernel Density Estimate curve
)
plt.title("Salary Distribution with KDE")
plt.show()

# 7️⃣ VIOLIN PLOT
sns.violinplot(
    x='department',  # x-axis categories
    y='salary',  # y-axis numerical variable
    hue='gender',  # split violins within each department by gender
    data=df,  # source DataFrame
    split=True,  # split violins for each hue level
    palette='muted'  # color palette
)
plt.title("Violin Plot of Salary Distribution by Department and Gender")
plt.show()

# 8️⃣ PAIR PLOT
sns.pairplot(
    df,  # source DataFrame
    hue='department',  # color points by department
    palette='Set1',  # color palette
    corner=True,  # plot only lower triangle
    diag_kind='kde',  # diagonal plots are Kernel Density Estimates
    kind='scatter',  # off-diagonal plots are scatter plots
    plot_kws={'alpha':0.6, 's':40},  # transparency and size of scatter points
    diag_kws={'shade':True}  # shade area under KDE curve
)
plt.suptitle("Pair Plot of Dataset", y=1.02)
plt.show()

# 9️⃣ REGRESSION PLOT (regplot)
sns.regplot(
    x='experience',  # x-axis numerical variable
    y='salary',  # y-axis numerical variable
    data=df,  # source DataFrame
    scatter_kws={'color':'blue'},  # set scatter point color
    line_kws={'color':'red'}  # set regression line color
)
plt.title("Regression Line of Experience vs Salary")
plt.show()