import nltk
import re
import contractions 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer , WordNetLemmatizer
from nltk import pos_tag,ngrams

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

data = {
    "text": [
        "I love this movie! It's amazing!",
        "Worst acting ever. Total waste of time.",
        "The plot was fine but too slow in the middle."`
    ],
    "label": ["positive", "negative", "neutral"]
}

df = pd.DataFrame(data)
print(df.head())
stopwords = set(stopwords.words("english"))
ps = PorterStemmer()
l = WordNetLemmatizer()

def pre(text):
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]'," ",text)
    text = text.lower().split()
    words = [l.lemmatize (word) for word in text if word not in stopwords]
    return " ".join(words)

def n_grams(text,n):
    text = text.split()
    grams = list(ngrams(text,n))
    print(grams)

df["clean"] = df["text"].apply(pre)

print(df)


for sentence in df["clean"]:
    n_grams(sentence, 2)

def get_pos_tags(text):
    tokens = text.split()
    print(pos_tag(tokens))

df["postags"] = df["clean"].apply(get_pos_tags)

pos_text = " ".join(df[df["label"]=="positive"]["clean"].tolist())
print(pos_text)

pos_w = WordCloud().generate(pos_text)
plt.imshow(pos_w, interpolation="bilinear")
plt.show()


