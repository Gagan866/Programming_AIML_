import requests  

from bs4 import BeautifulSoup

r = requests.get('https://www.nytimes.com/interactive/2017/06/23/opinion/trumps-lies.html')
print(r.text[:500])

soup = BeautifulSoup(r.text,"html.parser")

results = soup.find_all('span', attrs={'class':'short-desc'})

print(len(results))

print(results[0:3])

first_result = results[0]
print(first_result)
print(first_result.find('strong'))
print(first_result.find('strong').text)


records = []
for result in results:
    date = result.find('strong').text[0:-1] + ', 2017'
    lie = result.contents[1][1:-2]
    explanation = result.find('a').text[1:-1]
    url = result.find('a')['href']
    records.append((date, lie, explanation, url))


print(len(records))
print(records[0:3])



import pandas as pd
df = pd.DataFrame(records, columns=['date', 'lie', 'explanation', 'url'])
print(df)
