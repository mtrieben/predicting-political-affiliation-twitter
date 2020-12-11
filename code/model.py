import resource
import gc
import torch
import pickle
import json
import numpy as np
from torch.autograd import Variable
from preprocessing.generate_vocab import tokenize
from torch.utils.data import Dataset

# 
# Bidirectional GRU with Trainable Embeddings Model
# 
class Net(torch.nn.Module):
    def __init__(self, weights, maxlen):
        super(Net, self).__init__()
        # Hyperparameters
        self.hidden_size = 128
        self.feature_size = 300
        self.keep_size = 0.67
        self.maxlen = maxlen
        # Layers in our architecture 
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=1,
                                bidirectional=True, dropout=self.keep_size)
        self.dense1 = torch.nn.Linear(2*self.hidden_size, 1)
        self.dense2 = torch.nn.Linear(self.maxlen, 1)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=False)

    #  Forward pass of our bidirectional GRU model to produce logits
    def forward(self, x, hidden):
        x = self.embedding(x)
        gru_out, next_hidden = self.gru(x, hidden)
        logits = self.dense1(gru_out) + 1e-8
        logits = torch.squeeze(logits)
        logits = self.dense2(logits)
        return torch.sigmoid(logits), next_hidden


# Trains the model on training data
def train(model, dataset, max_len):
    
    # Adam Optimizer and Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    model.train()
    # Train for # of epochs
    for epoch in range(10):
        print("epoch", epoch)
        hidden = torch.zeros(2, max_len, model.hidden_size)
        count = 0
        for inputs, labels in dataset:
            model.zero_grad()
            optimizer.zero_grad()
            outputs, _ = model(inputs, hidden)
            outputs = outputs[:, -1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10 == 0:
                print("count:", count)

# Test the model on testing data
def test(model, dataset, max_len):
    hidden = torch.zeros(2, max_len, model.hidden_size)
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in dataset:
        outputs, _ = model(inputs, hidden)
        labels = (torch.eq(labels, 1.0))*1
        outputs = (torch.gt(outputs, 0.5))*1
        outputs = torch.squeeze(outputs)
        # Calculate and return accuracy
        total += len(inputs)
        correct += float((labels == outputs).sum())
    return correct / total


# Tokenizes/prepares the training/testing data
def prep_data(inputs_train, labels_train, inputs_test, labels_test, vocab):
    # Turn training labels into one hot
    labels_array = np.load(labels_train, allow_pickle=True)
    labels_final_train = [1 if x == 'Democrat' else 0 for x in labels_array]

    # Turn testing labels into one hot
    labels_array = np.load(labels_test, allow_pickle=True)
    labels_final_test = [1 if x == 'Democrat' else 0 for x in labels_array]

    # Tokenize and pad inputs
    # Training
    inputs_array_train = np.load(inputs_train, allow_pickle=True)
    inputs_array_train = [tokenize(x) for x in inputs_array_train]
    # Testing
    inputs_array_test = np.load(inputs_test, allow_pickle=True)
    inputs_array_test = [tokenize(x) for x in inputs_array_test]
    # Calculate max # of tokens out of training and testing
    max_len_train = np.array([len(x) for x in inputs_array_train])
    max_len_train = np.ndarray.max(max_len_train)
    max_len_test = np.array([len(x) for x in inputs_array_test])
    max_len_test = np.ndarray.max(max_len_test)
    max_len = max(max_len_train, max_len_test)
    # Pad and <UNK> training and testing to this max_length
    inputs_array_train = [x if len(
        x) >= max_len else x + ["<PAD>"]*(max_len - len(x)) for x in inputs_array_train]
    inputs_array_train = [
        [x if x in vocab else "<UNK>" for x in y] for y in inputs_array_train]
    inputs_array_train = [[vocab[x] for x in y] for y in inputs_array_train]
    inputs_array_test = [x if len(
        x) >= max_len else x + ["<PAD>"]*(max_len - len(x)) for x in inputs_array_test]
    inputs_array_test = [
        [x if x in vocab else "<UNK>" for x in y] for y in inputs_array_test]
    inputs_array_test = [[vocab[x] for x in y] for y in inputs_array_test]

    return inputs_array_train, inputs_array_test, labels_final_train, labels_final_test, max_len


if __name__ == "__main__":
    # Load data, vocab, and GLoVe embeddings
    pkl_file = open('data/vocab.pkl', 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    inputs_train, inputs_test, labels_train, labels_test, max_len = prep_data(
        "data/train/text.npy", "data/train/labels.npy", "data/test/text.npy", "data/test/labels.npy", vocab)
    glove = torch.Tensor(np.load("data/glove.npy", allow_pickle=True))
    # Create DataLoader for training data
    inputs_tensor = torch.LongTensor(inputs_train)
    labels_tensor = torch.Tensor(labels_train)
    data = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    dataset = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    # Create DataLoader for testing data
    inputs_test_tensor = torch.LongTensor(inputs_test)
    labels_test_tensor = torch.Tensor(labels_test)
    testing = torch.utils.data.TensorDataset(
        inputs_test_tensor, labels_test_tensor)
    testing_dataset = torch.utils.data.DataLoader(
        testing, batch_size=256, shuffle=False)
    # Initialize Model
    model = Net(glove, max_len)
    # Pre-test to compare accuracy later
    print("Pre-Test!")
    accuracy = test(model, testing_dataset, max_len)
    print("accuracy 1:", accuracy)
    # Train model
    print("Training...")
    train(model, dataset, max_len)
    print("Training done!")
    # Save weights if desired
    torch.save(model, "models/model80acc.pt")
    # Test modela nd print accuracy
    print("Testing...")
    accuracy = test(model, testing_dataset, max_len)
    print("Final Testing done!")
    print("Accuracy 2:", accuracy)
