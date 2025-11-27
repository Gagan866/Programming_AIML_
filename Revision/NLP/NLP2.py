import nltk 
import emoji
import pandas as pd
import re, contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ngrams

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from wordcloud import WordCloud

import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# ---------------------------------------------------------
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("Revision/Data/nlp_reviews.csv")

stop_words = set(stopwords.words("english"))
lm = WordNetLemmatizer()

# ---------------------------------------------------------
# PREPROCESSING FUNCTION
# ---------------------------------------------------------
def pre(text):

    # Expand contractions (don't -> do not)
    text = contractions.fix(text)

    # Convert emojis → words (:smiling_face:)
    text = emoji.demojize(text)

    # Remove everything except letters + spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Lemmatize + remove stopwords
    words = [
        lm.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean"] = df["Review"].apply(pre)
print(df[["Review", "clean"]])

# ---------------------------------------------------------
# POS TAGGING (OPTIONAL FEATURE)
# ---------------------------------------------------------
df["pos_tags"] = df["clean"].apply(lambda x: pos_tag(word_tokenize(x)))
print(df[["clean", "pos_tags"]])

# ---------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean"], df["Sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["Sentiment"]
)

# ---------------------------------------------------------
# VECTORIZATION – COUNT VECTORIZER
# ---------------------------------------------------------
vectorizer = CountVectorizer(max_features=3000)

# Fit only on TRAIN
X_train = vectorizer.fit_transform(X_train_raw)

# Transform TEST
X_test = vectorizer.transform(X_test_raw)

# ---------------------------------------------------------
# RANDOM FOREST WITH GRID SEARCH (GRID)
# ---------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("🔥 Best Parameters:", grid.best_params_)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n🔥 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - RandomForest + GridSearch")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# ---------------------------------------------------------
# WORDCLOUD FOR ENTIRE CLEANED TEXT
# ---------------------------------------------------------
all_text = " ".join(df["clean"])

wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="viridis"
).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - All Reviews", fontsize=16)
plt.show()

# ---------------------------------------------------------
# WORDCLOUD FOR POSITIVE REVIEWS
# ---------------------------------------------------------
pos_text = " ".join(df[df["Sentiment"] == 1]["clean"])

wc_pos = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Greens"
).generate(pos_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Positive Reviews", fontsize=16)
plt.show()

# ---------------------------------------------------------
# WORDCLOUD FOR NEGATIVE REVIEWS
# ---------------------------------------------------------
neg_text = " ".join(df[df["Sentiment"] == 0]["clean"])

wc_neg = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="Reds"
).generate(neg_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Negative Reviews", fontsize=16)
plt.show()
