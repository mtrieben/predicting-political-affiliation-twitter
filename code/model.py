import torch
import json

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_size = 128
        self.feature_size = 300
        self.keep_size = 0.67
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=1, 
        bidirectional=True, dropout=self.keep_size)
        self.dense = torch.nn.Linear(self.feature_size, 1)
        self.embedding = torch.nn.Embedding(self.vocab_size, self.feature_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        gru_out, next_hidden = self.gru(x, hidden)
        logits = self.dense(gru_out) + 1e-8
        return torch.sigmoid(logits), next_hidden

def train(model, dataset):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0436)
    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    for epoch in range(5):
        hidden = torch.zeros(1, 1, model.hidden_size)
        for i, data in enumerate(dataset):
            inputs, labels = data
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            labels = torch.equal(labels, 1.0)
            outputs = torch.gt(outputs, 0.5)
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





if __name__ == "__main__":
    with open("data/input.json") as f:
        inputs = json.load(f)
    with open("data/labels.json") as f:
        labels = json.load(f)
    data = torch.utils.data.TensorDataset(inputs, labels)
    dataset = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    model = Net()
    train(model, dataset)
    with open("data/testing.json") as f:
        testing = json.load(f)
    testing = torch.utils.data.TensorDataset(testing)
    testing_dataset = torch.utils.data.DataLoader(testing, batch_size=64, shuffle=False)
    accuracy = test(model, testing_dataset)
    print(accuracy)
