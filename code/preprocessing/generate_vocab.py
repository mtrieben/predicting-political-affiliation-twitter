import string
import numpy as np
import pickle


# Tokenizes a tweet
def tokenize(text):
    words = text.split()
    # For each word, replace with token if its a link or @
    for i in range(len(words)):
        w = words[i]
        if "http" in w:
            words[i] = "<LINK>"
        elif w.startswith("@"):
            words[i] = "<MENTION>"
        elif w.startswith(".@"):
            words[i] = "<MENTION>"
        else:
            # Remove punctuation
            words[i] = w.strip().strip(string.punctuation).strip(
                '…’‘”“ ,.•——?​​​​"\'').lower()
    return words

# Generate and save GLoVe embeddings for our vocabulary
def generate_glove_weights(vocab):
    embeddings_index = {}
    with open("data/glove.6B.300d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocab), 300))
    hits = 0.0
    for i, word in enumerate(vocab):
        if word.lower() in embeddings_index:
            hits += 1
            embedding_matrix[i] = embeddings_index[word.lower()]
        else:
            # If word does not have pretrained GLoVe embedding, randomly initialize an embedding for it
            embedding_matrix[i] = np.random.randn(1, 300)
    print("Word hit rate %f" % (hits/len(vocab)))
    return embedding_matrix


if __name__ == "__main__":
    # Initial vocab with token s
    vocab = {
        "<PAD>": 1000,
        "<UNK>": 1000,
        "<LINK>": 1000,
        "<MENTION>": 1000
    }

    index = 4

    training_tweets = np.load('data/train/text.npy', allow_pickle=True)
    testing_tweets = np.load('data/test/text.npy', allow_pickle=True)

    print(training_tweets.shape)
    print(testing_tweets.shape)

    # Add training Tweets to vocab
    for tweet in training_tweets:
        tweet = tokenize(tweet)
        for word in tweet:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Add testing Tweets to vocab
    for tweet in testing_tweets:
        tweet = tokenize(tweet)
        for word in tweet:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    print(len(vocab))

    words_to_unk = set()

    # Remove words that appear fewer than 6 times
    index = 0
    for word in vocab:
        if vocab[word] > 5:
            vocab[word] = index
            index += 1
        else:
            words_to_unk.add(word)

    for word in words_to_unk:
        del vocab[word]

    print(len(vocab))

    with open("data/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    glove = generate_glove_weights(vocab)
    np.save("data/glove.npy", glove)
