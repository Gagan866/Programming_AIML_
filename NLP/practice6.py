import nltk
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

sentence = "Barack Obama was born in Hawaii."

# Tokenize and tag POS
tokens = nltk.word_tokenize(sentence)


# Named entity chunking
entitys = nltk.ne_chunk(nltk.pos_tag(tokens))
print(entitys)