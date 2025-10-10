import pandas as pd
import nltk
import re, contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

data = pd.read_csv("CIEs/amazon_alexa.tsv", sep='\t')
print(data.head())
 
data.dropna(inplace=True)

l = WordNetLemmatizer()

def clean_text(text):
    text = contractions.fix(text)
    # print("Contraction : \n",text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    # print("RESUB : \n",text)
    text = text.lower()
    # print("LOWER : \n",text)
    words = text.split()
    # print("Split : \n",words)
    words = words = [l.lemmatize(word) for word in words if word not in STOPWORDS]
    # print("Lemmatization : \n",words)
    return " ".join(words)

data['cleaned_review'] = data['verified_reviews'].apply(clean_text)

x = data['cleaned_review']
y = data['feedback']

cv = CountVectorizer(max_features=5000)

x = cv.fit_transform(x).toarray()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

