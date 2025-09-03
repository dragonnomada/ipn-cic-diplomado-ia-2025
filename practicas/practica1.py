# pip install nltk==3.8.1
import nltk
print(nltk.__version__)
# Download the punkt tokenizer model (if you haven't already)
nltk.download('punkt')

from nltk.tokenize import word_tokenize
# Sample text
text = "Tokenization helps models understand language."
# Tokenizing text at the word level
word_tokens = word_tokenize(text)
print("Word-Level Tokens:", word_tokens)