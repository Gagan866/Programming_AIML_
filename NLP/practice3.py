import spacy
import nltk
import re

from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



STOPWORDS = set(stopwords.words('english'))

text = "Natural Language Processing (NLP) is a fascinating field of study."

text = re.sub('[^a-zA-Z]', ' ', text)
print("_"*40)
print("Retaining only text : \n",text)
print("_"*40)

tokens = text.lower().split()
print("_"*40)
print("Tokenization : \n",tokens)
print("_"*40)

cleaned_text = [word for word in tokens if word not in STOPWORDS]
print("_"*40)
print("After removing stopwords : \n",cleaned_text)
print("_"*40)

postagged = nltk.pos_tag(cleaned_text)
print("_"*40)
print("POS tagging : \n",postagged) 
print("_"*40)


