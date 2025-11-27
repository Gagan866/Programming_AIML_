import pandas as pd
import contractions
import emoji
import re

import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

df = pd.read_csv("nlp_data.csv")

print(df)

stop_words = set(stopwords.words("english"))

lm = WordNetLemmatizer()

def pre(text):
    text = contractions.fix(text)
    text = emoji.demojize(text)
    text = re.sub(r"[^a-zA-Z\s]"," ",text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lm.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def ngram(text,n):
    text = word_tokenize(text)
    ng = list(ngrams(text,n))
    return (ng)

def post(text):
    text = word_tokenize(text)
    pos = list(pos_tag(text))
    return (pos)

df["clean"] = df["Review"].apply(pre)
print(df)

df["ngrams"] = df["clean"].apply(lambda x : ngram(x,2))    
print(df)

df["postags"] = df["clean"].apply(lambda x : post(x))
print(df)

enc = {
    "Positive":1,
    "Neutral":0,
    "Negative":2
}

df["Sentiment"] = df["Sentiment"].map(enc)

x = df["clean"]
y = df["Sentiment"]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=42,stratify=y)

vec = CountVectorizer()

xtrain_vec = vec.fit_transform(xtrain)
xtest_vec = vec.transform(xtest)

log = LogisticRegression(max_iter=1000)
log.fit(xtrain_vec, ytrain)
log_pred = log.predict(xtest_vec)

print("\n====== Logistic Regression ======")
print("Accuracy:", accuracy_score(ytest, log_pred))
print(classification_report(ytest, log_pred))

# -------------------------------------------------------
# Random Forest Classifier
# -------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(xtrain_vec, ytrain)
rf_pred = rf.predict(xtest_vec)

print("\n====== Random Forest ======")
print("Accuracy:", accuracy_score(ytest, rf_pred))
print(classification_report(ytest, rf_pred))

wc = WordCloud(width=800, height=400, background_color='white')

sent_map = {1: "Positive", 0: "Neutral", 2: "Negative"}

for s in [1, 0, 2]:
    text = " ".join(df[df["Sentiment"] == s]["clean"])
    plt.figure(figsize=(10, 5))
    plt.imshow(wc.generate(text))
    plt.title(f"Word Cloud - {sent_map[s]}")
    plt.axis("off")
    plt.show()