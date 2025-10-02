# from bs4 import BeautifulSoup
# import requests
# import pandas as pd

# url = "https://www.simplilearn.com/top-technology-trends-and-jobs-article"

# response = requests.get(url)

# soup = BeautifulSoup(response.content, 'html')

# print(soup)

# print(soup.find_all("div",class_="main-content"))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text) 
    # Retain only text data
    text = re.sub('[^a-zA-Z]', ' ', text) 
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    words = text.split()
    # Remove stopwords
    return " ".join([word for word in words if word not in STOPWORDS])

def word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def main():

    user_input = input("1 for manual \n2 for automated from file \nEnter your choice: ")
    if user_input == '1':
        text = input("Enter your text: ")
        processed_text = preprocess_text(text)
        print("_"*40)
        print("Original Text: \n", text)
        print("_"*40)
        print("Preprocessed Text: \n", processed_text)
        print("_"*40)
        word_cloud(processed_text)
    
    elif user_input == '2':
        file_name = 'NLP/Data/articles.txt'
        txt = read_file(file_name)
        print("_"*40)
        print("Original Text: \n", txt) 
        print("_"*40)
        txt = preprocess_text(txt)
        print("_"*40)
        print("Preprocessed Text: \n", txt)
        print("_"*40)
        word_cloud(txt)
        
    else:
        print("Invalid input. Please enter 1 or 2.")

if __name__ == "__main__":
    main()