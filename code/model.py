import torch
import pickle
import json
import numpy as np
from torch.autograd import Variable
from preprocessing.generate_vocab import tokenize
from torch.utils.data import Dataset
import gc
import resource


class Net(torch.nn.Module):
    def __init__(self, weights, maxlen):
        super(Net, self).__init__()
        self.hidden_size = 128
        self.feature_size = 300
        self.keep_size = 0.67
        self.maxlen = maxlen
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=1,
                                bidirectional=True, dropout=self.keep_size)
        self.dense1 = torch.nn.Linear(2*self.hidden_size, 1)
        self.dense2 = torch.nn.Linear(self.maxlen, 1)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=False)

    def forward(self, x, hidden):
        x = self.embedding(x)
        gru_out, next_hidden = self.gru(x, hidden)
        logits = self.dense1(gru_out) + 1e-8
        logits = torch.squeeze(logits)
        logits = self.dense2(logits)
        return torch.sigmoid(logits), next_hidden


def train(model, dataset, max_len):

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0.0436)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1):
        print("epoch", epoch)
        hidden = torch.zeros(2, max_len, model.hidden_size)
        count = 0
        for inputs, labels in dataset:
            with torch.no_grad():
                optimizer.zero_grad()
                outputs, hidden = model(inputs, hidden)

                labels = (torch.eq(labels, 1.0))*1
                outputs = (torch.gt(outputs, 0.5))*1
                outputs = torch.squeeze(outputs)
                outputs = outputs.type(torch.DoubleTensor)
                labels = labels.type(torch.DoubleTensor)
                loss = criterion(outputs, labels)
                loss = Variable(loss, requires_grad=True)
                loss.backward()
                optimizer.step()
                count += 1
                if count % 10 == 0:
                    print("count:", count)
                    #     gc.collect()
                    #     print('maxrss = {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


def test(model, dataset, max_len):
    hidden = torch.zeros(2, max_len, model.hidden_size)
    correct = 0
    total = 0
    for inputs, labels in dataset:
        with torch.no_grad():
            outputs, hidden = model(inputs, hidden)
            labels = (torch.eq(labels, 1.0))*1
            outputs = (torch.gt(outputs, 0.5))*1
            outputs = torch.squeeze(outputs)
            total += len(inputs)
            correct += (labels == outputs).sum()
    print("total", total)
    print("correct", correct.item())
    return correct.item() / total


def prep_data(inputs_train, labels_train, inputs_test, labels_test, vocab):
    # Turn labels into one hot
    labels_array = np.load(labels_train, allow_pickle=True)
    labels_final_train = [1 if x == 'Democrat' else 0 for x in labels_array]

    # Turn labels into one hot
    labels_array = np.load(labels_test, allow_pickle=True)
    labels_final_test = [1 if x == 'Democrat' else 0 for x in labels_array]

    # Tokenize and pad inputs
    inputs_array_train = np.load(inputs_train, allow_pickle=True)
    inputs_array_train = [tokenize(x) for x in inputs_array_train]
    inputs_array_test = np.load(inputs_test, allow_pickle=True)
    inputs_array_test = [tokenize(x) for x in inputs_array_test]
    max_len_train = np.array([len(x) for x in inputs_array_train])
    max_len_train = np.ndarray.max(max_len_train)
    max_len_test = np.array([len(x) for x in inputs_array_test])
    max_len_test = np.ndarray.max(max_len_test)
    max_len = max(max_len_train, max_len_test)
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
    pkl_file = open('data/vocab.pkl', 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    inputs_train, inputs_test, labels_train, labels_test, max_len = prep_data(
        "data/train/text.npy", "data/train/labels.npy", "data/test/text.npy", "data/test/labels.npy", vocab)
    glove = torch.Tensor(np.load("data/glove.npy", allow_pickle=True))
    #data = NlpDataset(inputs, labels)
    inputs_tensor = torch.LongTensor(inputs_train[:30000])
    labels_tensor = torch.Tensor(labels_train[:30000])
    data = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    dataset = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    inputs_test_tensor = torch.LongTensor(inputs_test)
    labels_test_tensor = torch.Tensor(labels_test)
    testing = torch.utils.data.TensorDataset(
        inputs_test_tensor, labels_test_tensor)
    testing_dataset = torch.utils.data.DataLoader(
        testing, batch_size=64, shuffle=False)
    model = Net(glove, max_len)
    accuracy = test(model, testing_dataset, max_len)
    print("accuracy 1:", accuracy)
    print("Training...")
    train(model, dataset, max_len)
    print("Training done!")
    print("Testing...")
    accuracy = test(model, testing_dataset, max_len)
    print("Testing done!")
    print("accuracy 2:", accuracy)
