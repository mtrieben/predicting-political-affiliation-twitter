import numpy as np
import pandas as pd
import os.path

def save_dataset(dataset, path):
    text = dataset["full_text"].values
    np.save(os.path.join(path, "text.npy"), text)
    labels = dataset["party"].values
    np.save(os.path.join(path, "labels.npy"), labels)

data = pd.read_pickle("data/all_tweets.pkl")
data = data[pd.notnull(data["full_text"])]

np.random.seed(230)
shuffle_idx = np.random.permutation(len(data))
data_shuffled = data.iloc[shuffle_idx]

train = data_shuffled[:int(0.8*len(data))]
test = data_shuffled[int(0.8*len(data)):int(len(data))]

save_dataset(train, "data/train")
save_dataset(test, "data/test")