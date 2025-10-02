import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions

from wordcloud import WordCloud
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split,cross_val_predict,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

def read_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    print("_"*40)
    print("TSV Data : \n",df.head())
    print("_"*40)
    return df

def main():
    # Read text file
    file_name = 'NLP/Data/amazon_alexa.tsv'
    
    df_tsv = read_tsv(file_name)
    
    print("NUll Values in the dataset: \n", df_tsv.isnull().sum())
    
    df_tsv.dropna(inplace=True)
    
    print("Shape : \n", df_tsv.shape)
    print("Feedback column unique values: \n", df_tsv['feedback'].value_counts())
    
    df_tsv["cleaned"] = df_tsv["verified_reviews"].apply(preprocess_text)
    print("Cleaned Text : \n", df_tsv[["cleaned","verified_reviews"]].head(10))
    
    # Vectorization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_tsv['cleaned']).toarray()
    y = df_tsv['feedback'].values
    print("Vectorized Text : \n", X)
    print("Labels : \n", y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train shape: ", X_train.shape, y_train.shape)
    print("Test shape: ", X_test.shape, y_test.shape)
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # Positive reviews
    positive_reviews = " ".join(df_tsv[df_tsv['feedback'] == 1]['cleaned'])
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Reviews')
    plt.show()
    
    # Negative reviews
    negative_reviews = " ".join(df_tsv[df_tsv['feedback'] == 0]['cleaned'])
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Reviews')
    plt.show()
    
    # Example prediction
    example_review = "This is the best product I have ever used!"
    example_cleaned = [preprocess_text(example_review)]   
    example_vectorized = vectorizer.transform(example_cleaned).toarray()
    prediction = model.predict(example_vectorized)
    print(f"Review: '{example_review}' => Predicted Feedback: {prediction[0]}")
    print("Predicted Sentiment: ", "Positive" if prediction[0] == 1 else "Negative")
    
    test_review = ["This product is terrible and I want a refund.",
                   "I absolutely love this! It works perfectly.",
                   "Not worth the money.",
                   "Exceeded my expectations!"]
    
    for review in test_review:
        cleaned_review = [preprocess_text(review)]
        vectorized_review = vectorizer.transform(cleaned_review).toarray()
        pred = model.predict(vectorized_review)
        print(f"Review: '{review}' => Predicted Feedback: {pred[0]} => Sentiment: {'Positive' if pred[0] == 1 else 'Negative'}")
        
        
    
if __name__ == "__main__":
    main()