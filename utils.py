import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer for English words
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split a sentence into words or tokens.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Reduce a word to its root form (stem).
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Create a bag of words vector.
    """
    sentence_stems = [stem(w) for w in tokenized_sentence]
    bag = [1 if w in sentence_stems else 0 for w in words]
    return bag