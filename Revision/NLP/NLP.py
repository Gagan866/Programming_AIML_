import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

url = "https://www.engadget.com/reviews/wearables/"
stopwords = set(stopwords.words('english'))
nltk.download('punkt')  
request = requests.get(url)
lem = WordNetLemmatizer()

soup = BeautifulSoup(request.text,"html.parser")
titles = []

# Titles on Engadget often appear inside <h2> or <span>
for span in soup.find_all("span"):
    text = span.get_text(strip=True)
    if text:
        titles.append(text)

print(titles)

df = pd.DataFrame({
    "Title" : titles
})

print(df)

def preprocessing(text):
    text = text.lower()
    tokens = word_tokenize(text)
    clean_words = []
    for token in tokens:
        if token in stopwords or token.isnumeric() or token in string.punctuation:
            continue
        clean_words.append(lem.lemmatize(token))

    return " ".join(clean_words)


df['cleaned_title'] = df['Title'].apply(preprocessing)
print(df[['Title', 'cleaned_title']])


def auto_label(text):
    text = text.lower()

    positive_keywords = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'best', 'positive', 'satisfied',"beautiful"]

    negative_keywords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'negative', 'disappointed', 'poor', "ugly","slow","disappointing"]

    score = 0

    for word in positive_keywords:
        if word in text:
            score += 1
    for word in negative_keywords:
        if word in text:
            score -= 1
    return 1 if score > 0 else 0

df['label'] = df['cleaned_title'].apply(auto_label)
print(df[['Title', 'cleaned_title', 'label']])