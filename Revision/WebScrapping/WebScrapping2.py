import requests
from bs4 import BeautifulSoup
import pandas as pd

urls = [
    "https://www.bbc.com/",
    "https://www.bbc.com/sport",
    "https://www.bbc.com/business",
    "https://www.bbc.com/innovation"
]

category = [
    "General",
    "Sports",
    "Business",
    "Innovation"
]

headlines = []
h_title = []

i = 0 
for url in urls:
    url = requests.get(url)

    soup = BeautifulSoup(url.text,"html.parser")

    results = soup.find_all("h2")

    for r in results:
        title = r.text.strip()
        if title:
            headlines.append(title)
            h_title.append(category[i])
    i += 1

df1 = {
    "headlines" : headlines,
    "title" : h_title
}


df = pd.DataFrame(df1)
print(df)