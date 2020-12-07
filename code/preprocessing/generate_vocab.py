import string
import numpy as np
import pickle


def tokenize(text):
    words = text.split()
    for i in range(len(words)):
        w = words[i]
        if "http" in w:
            words[i] = "<LINK>"
        elif w.startswith("@"):
            words[i] = "<MENTION>"
        elif w.startswith(".@"):
            words[i] = "<MENTION>"
        else:
            words[i] = w.strip().strip(string.punctuation).strip(
                '…’‘”“ ,.•——?​​​​"\'').lower()
    return words


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
            # print(word)
            embedding_matrix[i] = np.random.randn(1, 300)
    print("Word hit rate %f" % (hits/len(vocab)))
    return embedding_matrix


if __name__ == "__main__":
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<LINK>": 2,
        # "<HASHTAG>": 3,
        "<MENTION>": 3
    }

    index = 4

    training_tweets = np.load('data/train/text.npy', allow_pickle=True)
    testing_tweets = np.load('data/test/text.npy', allow_pickle=True)

    print(training_tweets.shape)
    print(testing_tweets.shape)

    single_uses = set()

    for tweet in training_tweets:
        tweet = tokenize(tweet)
        for word in tweet:
            if word not in vocab:
                vocab[word] = index
                index += 1
                single_uses.add(word)
            else:
                if word in single_uses:
                    single_uses.remove(word)

    for tweet in testing_tweets:
        tweet = tokenize(tweet)
        for word in tweet:
            if word not in vocab:
                vocab[word] = index
                index += 1
                single_uses.add(word)
            else:
                if word in single_uses:
                    single_uses.remove(word)

    print(len(vocab))

    for word in single_uses:
        del vocab[word]

    print(len(vocab))

    with open("data/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    glove = generate_glove_weights(vocab)
    np.save("data/glove.npy", glove)
