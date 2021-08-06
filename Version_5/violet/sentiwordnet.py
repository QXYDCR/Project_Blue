import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

sentence = "One of the best movie of all time. Period."


# Removing Punctuations
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
for x in sentence:
  if x in punctuations:
    sentence = sentence.replace(x, "")

print(sentence)

# Change Case + Tokenization

Tokens = nltk.word_tokenize(sentence.lower())
print(Tokens)















