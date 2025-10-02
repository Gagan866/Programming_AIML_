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

def generate_char_ngrams(text, n):
    # Generate character n-grams
    n_grams = []
    for i in range(len(text)-n+1):
        n_grams.append(text[i:i+n])
    print("_"*40)
    print(f"Generated character {n}-grams : \n",n_grams)
    print("_"*40)

def Stemmer(words):
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in words]
    print("_"*40)
    print("Stemmed words : \n",stemmed_words)
    print("_"*40)
    return stemmed_words

def lemmatize_word(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    print("_"*40)
    print("Lemmatized words : \n",lemmatized_words)
    print("_"*40)
    return lemmatized_words

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

def find_pos_tag(word, pos_tags):
    for w, tag in pos_tags:
        if w == word:
            return tag
    return 'N/A'

def write_file_txt(file_path, unique_words, stemmed_words, pos_tags_stemmed, lemmatized_words, pos_tags_lemmatized, df_postags):
    with open(file_path, 'w') as file:
        file.write("Unique Word\tStemmed Word\tPOS Tag_Description\tLemmatized Word\tPOS Tag_Description\n")
        for i,word in enumerate(unique_words):
            
            stemmed = stemmed_words[i] 
            pos_stemmed = find_pos_tag(stemmed, pos_tags_stemmed)

            lemmatized = lemmatized_words[i]
            pos_lemmatized = find_pos_tag(lemmatized, pos_tags_lemmatized)

            description_stemmed = df_postags.loc[df_postags['Tag'] == pos_stemmed, 'Description'].values
            description_stemmed = description_stemmed[0] if len(description_stemmed) > 0 else 'N/A'
            description_lemmatized = df_postags.loc[df_postags['Tag'] == pos_lemmatized, 'Description'].values
            description_lemmatized = description_lemmatized[0] if len(description_lemmatized) > 0 else 'N/A'
            
            file.write(f"{word}\t{stemmed}\t{description_stemmed}\t{lemmatized}\t{description_lemmatized}\n")

def write_file_csv(file_path, unique_words, stemmed_words, pos_tags_stemmed, lemmatized_words, pos_tags_lemmatized, df_postags):
    data = []
    for i,word in enumerate(unique_words):
        
        stemmed = stemmed_words[i] 
        pos_stemmed = find_pos_tag(stemmed, pos_tags_stemmed)

        lemmatized = lemmatized_words[i]
        pos_lemmatized = find_pos_tag(lemmatized, pos_tags_lemmatized)

        description_stemmed = df_postags.loc[df_postags['Tag'] == pos_stemmed, 'Description'].values
        description_stemmed = description_stemmed[0] if len(description_stemmed) > 0 else 'N/A'
        description_lemmatized = df_postags.loc[df_postags['Tag'] == pos_lemmatized, 'Description'].values
        description_lemmatized = description_lemmatized[0] if len(description_lemmatized) > 0 else 'N/A'
        
        data.append([word, stemmed, description_stemmed, lemmatized, description_lemmatized])
    
    df = pd.DataFrame(data, columns=["Unique Word", "Stemmed Word", "POS Tag_Description", "Lemmatized Word", "POS Tag_Description"])
    df.to_csv(file_path, index=False)

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
    
    stemmed_words = Stemmer(unique_words)
    lemmatized_words = lemmatize_word(unique_words)
    
    post_tags_stemmed = show_pos_tags(' '.join(stemmed_words))
    post_tags_lemmatized = show_pos_tags(' '.join(lemmatized_words))
    
    df_postags = read_csv_postags()
    
    write_file_txt('NLP/TextFiles/processed_text.txt', unique_words, stemmed_words, post_tags_stemmed, lemmatized_words,post_tags_lemmatized, df_postags)
    write_file_csv('NLP/TextFiles/processed_text.csv', unique_words, stemmed_words, post_tags_stemmed, lemmatized_words,post_tags_lemmatized, df_postags)

if __name__ == "__main__":
    main()