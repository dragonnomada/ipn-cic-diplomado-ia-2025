from tokenizers import BertWordPieceTokenizer

# Sample corpus as a list of sentences
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# Write corpus to a temporary text file for training
with open("corpus.txt", "w") as f:
    for line in corpus:
        f.write(f"{line}\n")

# Initialize the BPE tokenizer
tokenizer = BertWordPieceTokenizer()

# Train the tokenizer on the corpus
tokenizer.train(files=["corpus.txt"], vocab_size=100, min_frequency=1)

# Save the tokenizer
tokenizer.save_model(".", "wordpiece_tokenizer")

# Test the tokenizer
encoded = tokenizer.encode("understand")
print("Encoded:", encoded.tokens)

# Decode tokens back to text
decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)