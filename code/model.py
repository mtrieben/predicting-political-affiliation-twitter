import torch
import pickle
import json
import numpy as np
from preprocessing.generate_vocab import tokenize
from torch.utils.data import Dataset

class Net(torch.nn.Module):
    def __init__(self, weights):
        super(Net, self).__init__()
        self.hidden_size = 128
        self.feature_size = 300
        self.keep_size = 0.67
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=1, 
        bidirectional=True, dropout=self.keep_size)
        self.dense = torch.nn.Linear(2*self.hidden_size, self.feature_size)
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=False)

    def forward(self, x, hidden):
        x = self.embedding(x)
        gru_out, next_hidden = self.gru(x, hidden)
        logits = self.dense(gru_out) + 1e-8
        return torch.sigmoid(logits), next_hidden

class NlpDataset(Dataset):
    def __init__(self, inputs, labels):
        self.samples = []
        for i in range(len(inputs)):
            self.samples.append((inputs[i], labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1]

def train(model, dataset):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0436)
    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    for epoch in range(5):
        print(epoch)
        hidden = torch.zeros(2, 59, model.hidden_size)
        for inputs, labels in dataset:
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            labels = (torch.eq(labels, 1.0))*1
            outputs = (torch.gt(outputs, 0.5))*1
            loss += criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, dataset):
    hidden = torch.zeros(1, 1, model.hidden_size)
    correct = 0
    total = 0
    for i, data in enumerate(dataset):
        inputs, labels = data
        outputs, hidden = model(inputs, hidden)
        labels = torch.equal(labels, 1.0)
        outputs = torch.gt(outputs, 0.5)
        total += len(inputs)
        correct += (labels == outputs).sum()
    return correct / total

def prep_data(inputs, labels, vocab):
    # Turn labels into one hot
    labels_array = np.load(labels, allow_pickle=True)
    labels_final = [1 if x == 'Democrat' else 0 for x in labels_array]

    # Tokenize and pad inputs
    inputs_array = np.load(inputs, allow_pickle=True)
    inputs_array = [tokenize(x) for x in inputs_array]
    max_len = np.array([len(x) for x in inputs_array])
    max_len = np.ndarray.max(max_len)
    print(max_len)
    inputs_array = [x if len(x) >= max_len else x + ["<PAD>"]*(max_len - len(x)) for x in inputs_array]
    inputs_array = [[vocab[x] for x in y] for y in inputs_array]
    return inputs_array, labels_final


if __name__ == "__main__":
    pkl_file = open('../data/vocab.pkl', 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    inputs, labels = prep_data("../data/train/text.npy", "../data/train/labels.npy", vocab)
    glove = torch.Tensor(np.load("../data/glove.npy", allow_pickle=True))
    #data = NlpDataset(inputs, labels)
    #print(inputs)
    inputs_tensor = torch.LongTensor(inputs)
    labels_tensor = torch.Tensor(labels)
    data = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    dataset = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    model = Net(glove)
    train(model, dataset)
    # inputs_test, labels_test = prep_data("../data/test/text.npy", "../data/test/labels.npy")
    # testing = torch.utils.data.TensorDataset(testing)
    # testing_dataset = torch.utils.data.DataLoader(testing, batch_size=64, shuffle=False)
    # accuracy = test(model, testing_dataset)
    # print(accuracy)
