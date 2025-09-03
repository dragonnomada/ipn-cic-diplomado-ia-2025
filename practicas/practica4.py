import sentencepiece as spm

# Create a sample corpus file
corpus = ["hug", "pug", "pun", "bun", "hugs"]
with open("corpus.txt", "w") as f:
    for word in corpus:
        f.write(f"{word}\n")

# Step 1: Train a Unigram tokenizer
spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="unigram_tokenizer",
    vocab_size=13,  # Adjust based on desired vocabulary size
    model_type="unigram"
)

# Step 2: Load the trained Unigram model
sp = spm.SentencePieceProcessor(model_file="unigram_tokenizer.model")

# Step 3: Tokenize text with the Unigram model
words = ["hug", "pug", "pun", "hugs", "bun"]
for word in words:
    tokens = sp.encode(word, out_type=str)
    print(f"Word: {word} -> Tokens: {tokens}")

# Step 4: Decode tokens back to text
for word in words:
    token_ids = sp.encode(word)
    decoded_word = sp.decode(token_ids)
    print(f"Tokens: {token_ids} -> Decoded word: {decoded_word}")