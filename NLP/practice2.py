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

    # Remove stopwords and stemming
    # ps = PorterStemmer()
    # processed_words = [ps.stem(word) for word in words if word not in STOPWORDS]
    # print("_"*40)
    # print("Processed words : \n",processed_words)
    # print("_"*40)
    
    return [word for word in words if word not in STOPWORDS]

def generate_word_ngrams(words, n): 
    # Generate n-grams
    n_grams = ngrams(words, n)
    n_grams_list = list(n_grams)
    print("_"*40)
    print(f"Generated {n}-grams : \n",n_grams_list)
    print("_"*40)

def generate_char_ngrams(text, n):
    # Generate character n-grams
    n_grams = []
    for i in range(len(text)-n+1):
        n_grams.append(text[i:i+n])
    print("_"*40)
    print(f"Generated character {n}-grams : \n",n_grams)
    print("_"*40)

def lemmatize_word():
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words_pos = {"running": "v", "better": "a", "cats": "n", "studies": "n", "fairly": "r","studying": "v"}
    for word, pos in words_pos.items():
        stemmed_word = ps.stem(word)
        lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
        print(f"Word: {word}, Stemmed: {stemmed_word}, Lemmatized: {lemmatized_word}")

def show_pos_tags(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    print("_"*40)   
    print("Part-of-Speech Tags : \n",pos_tags)
    print("_"*40)
    return pos_tags

    

def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def write_file(file_path, pos_tags, df_postags):
    with open(file_path, 'w') as file:
        file.write("Word\tPOS Tag\tDescription\n")
        for word, tag in pos_tags:
            desc_row = df_postags[df_postags['Tag'] == tag]
            if not desc_row.empty:
                description = desc_row.iloc[0]['Description']
            else:
                description = "N/A"
            file.write(f"{word}\t{tag}\t{description}\n")

def read_csv_postags():
    df = pd.read_csv('NLP/TextFiles/pos_tags.csv')
    print("_"*40)
    print("CSV Data : \n",df.head())
    print("_"*40)
    return df

def main():
    file_path = 'NLP/TextFiles/postags.txt'
    text = read_file(file_path)

    processed_words = preprocess_text(text)
    
    # Uncomment below to use n-gram generation interactively
    # n_word = int(input("Enter the value of n for n-grams words (e.g., 2 for bigrams, 3 for trigrams): ").strip())
    # generate_word_ngrams(processed_words, n_word)
    
    processed_text = ''.join(processed_words)
    
    # n_char = int(input("Enter the value of n for n-grams characters (e.g., 2 for bigrams, 3 for trigrams): ").strip())
    # generate_char_ngrams(processed_text, n_char)
    
    # lemmatize_word()

    unique_words = set(processed_words)
    print("_"*40)
    print("Unique words after removing stopwords : \n",unique_words)
    print("_"*40)
    post_tags = show_pos_tags(' '.join(unique_words))
    df_postags = read_csv_postags()
    write_file('NLP/TextFiles/processed_text.txt', post_tags, df_postags)

if __name__ == "__main__":
    main()