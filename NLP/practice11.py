import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams

from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    # Retain only text data
    text = re.sub('[^a-zA-Z]', ' ', text)
    print("_"*40)
    print("Retaining only text : \n",text)
    print("_"*40)
    
    # Convert to lowercase
    text = text.lower()
    print("_"*40)
    print("Converting to lowercase : \n",text)
    print("_"*40)

    # Tokenization
    words = text.split()
    print("_"*40)
    print("Tokenization : \n",words)
    print("_"*40)
    
    # Remove stopwords
    return [word for word in words if word not in STOPWORDS]


def generate_word_ngrams(words, n): 
    # Generate n-grams
    n_grams = ngrams(words, n)
    n_grams_list = list(n_grams)
    print("_"*40)
    print(f"Generated {n}-grams : \n",n_grams_list)
    print("_"*40)
    
def main():
    file_path = 'NLP/TextFiles/postags.txt'
    text = "Machine Learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions relying on"
    
    # Uncomment below to use n-gram generation interactively
    n_word = int(input("Enter the value of n for n-grams words (e.g., 2 for bigrams, 3 for trigrams): ").strip())
    generate_word_ngrams(text.split(), n_word)
    
    preprocess_ = preprocess_text(text)
    generate_word_ngrams(preprocess_, n_word)
    
if __name__ == "__main__":
    main()