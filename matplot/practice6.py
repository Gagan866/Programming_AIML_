import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'Age': [1, 2, 3, 4, 5, 6, 7],
    'Price': [20000, 19500, 18000, 17500, 16000, 15000, 14000]
})

# Basic Plot
sns.regplot(x="Age", y="Price", data=df)
plt.title("Car Price vs Age")
plt.show()
sns.set_style("whitegrid")      # Adds grid on white background
sns.set_context("talk")         # Enlarges font & labels for presentation

sns.regplot(x="Age", y="Price", data=df,
            marker="*", ci=95,
            line_kws={"color": "black", "linewidth": 2},
            scatter_kws={"color": "red", "alpha": 0.6})

plt.title("Car Price vs Age", fontsize=18)
plt.xlabel("Car Age (Years)")
plt.ylabel("Price (in Rs)")
plt.grid(True)
plt.show()
styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
contexts = ["paper", "notebook", "talk", "poster"]

for style in styles:
    for context in contexts:
        sns.set_style(style)
        sns.set_context(context)

        sns.regplot(x="Age", y="Price", data=df)
        plt.title(f"Style: {style}, Context: {context}")
        plt.show()
