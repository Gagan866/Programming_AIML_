import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import re, contractions
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

data = pd.read_csv("NLP/Data/amazon_alexa.tsv", sep='\t')
print(data.head())

data.dropna(inplace=True)

l = WordNetLemmatizer()

def clean_text(text):
    text = contractions.fix(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [l.lemmatize (word) for word in text if word not in stopwords]
    return " ".join(words)

data['cleaned_review'] = data['verified_reviews'].apply(clean_text)

X = data['cleaned_review']
y = data['feedback']  # 1 = positive, 0 = negative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = CountVectorizer(max_features=5000)
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_cv, y_train)

y_pred = model.predict(X_test_cv)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

