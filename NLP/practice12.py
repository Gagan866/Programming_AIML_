import re
import nltk
import contractions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk import ngrams,pos_tag
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')


data = {
    "text": [
        "I love this movie! It's amazing!",
        "Worst acting ever. Total waste of time.",
        "The plot was fine but too slow in the middle."
    ],
    "label": ["positive", "negative", "neutral"]
}

df = pd.DataFrame(data)
print(df.head())


df["text"] = df["text"].apply(contractions.fix)
print("_"*40)
print("After expanding contractions : \n",df["text"])
print("_"*40)


df["text"] = df["text"].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

print("After removing non-alphabetic characters : \n",df["text"])

df["text"] = df["text"].str.lower()
print("_"*40)
print("After converting to lowercase : \n",df["text"])
print("_"*40)

df["tokens"] = df["text"].apply(word_tokenize)
print("After Tokenization : \n",df["tokens"])
print("_"*40)
STOPWORDS = set(stopwords.words('english'))
df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in STOPWORDS])
print("After removing stopwords : \n",df["tokens"])
print("_"*40)

lemmatizer = WordNetLemmatizer()
df["lemmatized"] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
print("After Lemmatization : \n",df["lemmatized"])
print("_"*40)

stemmer = PorterStemmer()
df["stemmed"] = df["tokens"].apply(lambda x: [stemmer.stem(word) for word in x])
print("After Stemming : \n",df["stemmed"])
print("_"*40)   

df["bigrams"] = df["tokens"].apply(lambda x: list(ngrams(x, 2)))
print("Bigrams : \n",df["bigrams"])
print("_"*40)

df["pos_tags"] = df["tokens"].apply(pos_tag)
print("POS Tags : \n",df["pos_tags"])
print("_"*40)
print("POS Tags of first entry : \n",df["pos_tags"].iloc[0])
print("_"*40)
print("POS Tags of second entry : \n",df["pos_tags"].iloc[1])
print("_"*40)   

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))



print("Classification Report:\n", classification_report(y_test, y_pred))


# WOrd Cloud
all_words = ' '.join([word for tokens in df['tokens'] for word in tokens])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7)) 
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()