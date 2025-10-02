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


df = pd.read_csv('NLP/Data/IMDB Dataset.csv', nrows=2000)

print("First 5 entries in the dataset: \n", df.head())
print("NUll Values in the dataset: \n", df.isnull().sum())

df["cleaned"] = df["review"].apply(preprocess_text)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
print("Cleaned Data: \n", df[['review', 'cleaned', 'sentiment']].head())

tfi = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
vectorizer = tfi
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['sentiment'].values
print("Vectorized Text : \n", X)
print("Labels : \n", y)

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
positive_reviews = " ".join(df[df['sentiment'] == 1]['cleaned'])
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Reviews')
plt.show()

# Negative reviews
negative_reviews = " ".join(df[df['sentiment'] == 0]['cleaned'])
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Negative Reviews')
plt.show()