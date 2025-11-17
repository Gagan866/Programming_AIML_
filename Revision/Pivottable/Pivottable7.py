import pandas as pd
import numpy as np

data = pd.date_range(start="1/1/2024",end="1/10/2024",freq="D")

print(data)

value = np.random.randint(10,100,size=len(data))

df = pd.DataFrame({"date":data,"value":value}).set_index("date")

print(df)

print(df.loc["2024-01-03":"2024-01-06"])
