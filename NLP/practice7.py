import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions

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
    # Original text
    print("_"*40)
    print("Original Text : \n",text)
    print("_"*40)
    
    # Expand contractions
    text = contractions.fix(text)
    print("_"*40)       
    print("Expanding Contractions : \n",text)
    print("_"*40)
    
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

def postag_words(words):
    pos_tags = nltk.pos_tag(words)
    print("_"*40)
    print("POS Tags : \n",pos_tags)
    print("_"*40)
    return words, pos_tags


def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


def read_csv_postags():
    df = pd.read_csv('NLP/TextFiles/pos_tags.csv')
    print("_"*40)
    print("CSV Data : \n",df.head())
    print("_"*40)
    return df

def write_csv(unique_words, stemmed_word, stemmed_postags, lemmatized_word, lemmatized_postags, df_postags):
    df_postags_description = df_postags['Description']

    df = []

    for unique_word, stemmed_word, stemmed_postag, lemmatized_word, lemmatized_postag in zip(unique_words, stemmed_word, stemmed_postags, lemmatized_word, lemmatized_postags):
        stemmed_description = df_postags_description[df_postags['Tag'] == stemmed_postag[1]].values
        lemmatized_description = df_postags_description[df_postags['Tag'] == lemmatized_postag[1]].values

        df.append([unique_word, stemmed_word, stemmed_postag[1], stemmed_description, lemmatized_word, lemmatized_postag[1], lemmatized_description])

    df = pd.DataFrame(df, columns=['Unique Word', 'Stemmed Word', 'Stemmed POS Tag', 'Stemmed Description', 'Lemmatized Word', 'Lemmatized POS Tag', 'Lemmatized Description'])    
    
    print("_"*40)
    print("Final DataFrame : \n",df)
    print("_"*40)
    
    df.to_csv('NLP/TextFiles/processed_text1.csv', index=False)
    
def postags_to_description(word_postag, df_postags):
    df_postags_description = df_postags['Description']
    for word, postag in zip(word_postag[0], word_postag[1]):
        description = df_postags_description[df_postags['Tag'] == postag[1]].values
        print(f"Word: {word}, POS Tag: {postag[1]}, Description: {description}")

def named_entity_recognition(words):

    # POS tagging
    pos_tags = nltk.pos_tag(words)
    
    # Perform NER
    named_entities = nltk.ne_chunk(pos_tags, binary=False)

    print("_"*40)
    print("Named Entities Tree : \n", named_entities)
    print("_"*40)

def vectorize_text(texts):
    # Bag of Words
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(texts)
    print("_"*40)
    print("Bag of Words Representation : \n", X_bow.toarray())
    print("Feature Names : \n", vectorizer.get_feature_names_out())
    print("_"*40)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    print("_"*40)
    print("TF-IDF Representation : \n", X_tfidf.toarray())
    print("Feature Names : \n", tfidf_vectorizer.get_feature_names_out())
    print("_"*40)
    

def main():
    file_path = 'NLP/TextFiles/postags.txt'
    text = read_file(file_path)
    
    ner_entities = named_entity_recognition(text.split())

    processed_words = preprocess_text(text)
    
    # Uncomment below to use n-gram generation interactively
    # n_word = int(input("Enter the value of n for n-grams words (e.g., 2 for bigrams, 3 for trigrams): ").strip())
    # generate_word_ngrams(processed_words, n_word)
    
    processed_text = ''.join(processed_words)
    
    # n_char = int(input("Enter the value of n for n-grams characters (e.g., 2 for bigrams, 3 for trigrams): ").strip())
    # generate_char_ngrams(processed_text, n_char)

    unique_words = set(processed_words)
    print("_"*40)
    print("Unique words after removing stopwords : \n",unique_words)
    print("_"*40)
    
    stemmed_words = Stemmer(unique_words)
    lemmatized_words = lemmatize_word(unique_words)

    stemmed_word, stemmed_postags = postag_words(stemmed_words)
    
    lemmatized_word, lemmatized_postags = postag_words(lemmatized_words)

    df_postags = read_csv_postags()

    write_csv(unique_words, stemmed_word, stemmed_postags, lemmatized_word, lemmatized_postags, df_postags)

    wts_app = preprocess_text("12/09/25, 8:30 PM - Riya: OMG 😂😂 exams coming!! Studying hard yaar.")
    print("WhatsApp Application Example : \n",wts_app)
    print("_"*40)
    wts_app_pos = postag_words(wts_app)
    print("_"*40)
    postags_to_description(wts_app_pos, df_postags)

    ex1 = preprocess_text("Wow!! NLP is amazing 🤩 #AIrocks Visit: https://openai.com for more info.")
    print("WhatsApp Application Example : \n",ex1)
    print("_"*40)
    ex1_pos = postag_words(ex1)
    print("_"*40)
    postags_to_description(ex1_pos, df_postags)
    
    ex2 = preprocess_text("Can't wait for the new season of my favorite show on Netflix! 🎬🍿 #Excited")
    print("WhatsApp Application Example : \n",ex2)
    print("_"*40)
    ex2_pos = postag_words(ex2)
    print("_"*40)
    postags_to_description(ex2_pos, df_postags)


    
    
if __name__ == "__main__":
    main()