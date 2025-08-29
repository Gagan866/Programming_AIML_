import matplotlib.pyplot as plt
import pandas as pd


dep = ["Edu","Health","Defence","Infra"]
allocaton = [25,30,20,25]
stream = ["Sci","Commerce","Arts","Diploma"]
enrol = [45,35,10,10]
job = ["Dev","QA","Designers","Admin"]
dist = [50,20,15,15]
meth = ["Credit","UPI","Cash","Wallets"]
use = [45,35,10,10]
disease = ["Viral","Bacterial","Chronic","Other"]
prop = [40,35,15,10]

data = {"Department":dep,"Allocation":allocaton}


df = pd.DataFrame(data)

print(df.head(10))
# explode = [0.1, 0.2, 0.3, 0.4]

# plt.title("Budget")
# plt.pie(x=allocaton,labels=dep,colors=["b","c","g"],autopct="%.f%%",wedgeprops={"edgecolor":"black","linewidth":1},shadow=True,startangle=90,explode=explode)
# plt.show()

# plt.title("Enrolement by stream")
# plt.pie(x=enrol,labels=stream,colors=["y","r","orange"],autopct="%.f%%",wedgeprops={"edgecolor":"black","linewidth":1},explode=explode,shadow=True)
# plt.show()

# plt.title("JOB")
# plt.pie(x=dist,labels=job,colors=["w","black","brown"],autopct="%.f%%",wedgeprops={"edgecolor":"black","linewidth":1},explode=explode,shadow=True)
# plt.show()

# plt.title("Payment")
# plt.pie(x=use,labels=meth,colors=["y","b","c"],autopct="%.f%%",wedgeprops={"edgecolor":"black","linewidth":1},explode=explode,shadow=True)
# plt.show()

# plt.title("Disease")
# plt.pie(x=prop,labels=disease,colors=["black","r","gold"],autopct="%.f%%",wedgeprops={"edgecolor":"black","linewidth":1},explode=explode,shadow=True)
# plt.show()