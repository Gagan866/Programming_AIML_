import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Add sentiment pipeline
nlp.add_pipe("spacytextblob")

# Example movie reviews
reviews = [
    "I absolutely loved the movie, great story and acting!",
    "It was boring and a waste of time.",
    "The movie was okay, not too bad but not great either."
]

for review in reviews:
    doc = nlp(review)
    print(f"Review: {review}")
    print(f"Polarity: {doc._.blob.polarity}, Subjectivity: {doc._.blob.subjectivity}\n")
