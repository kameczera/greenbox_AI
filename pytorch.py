import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(256 * 256, 5120),
            nn.ReLU(),
            nn.Linear(5120, 5120),
            nn.ReLU(),
            nn.Linear(5120, 5120),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits
    
model = DNN().to("cuda")

if __name__ == '__main__':
    X_train = sys.argv[1]
    Y_train = sys.argv[2]
    X_train = np.loadtxt(X_train, delimiter = ',').astype(np.float32) / 255.0
    Y_train = np.loadtxt(Y_train, delimiter = ',').astype(np.float32).reshape(1, -1)
    Y_train = Y_train.astype(np.float32)
    # Tem que dar o squeeze pq o resultado é uma matriz 1 x N
    Y_train = Y_train.squeeze()
    X_train = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    Y_train = torch.tensor(Y_train, dtype=torch.long).to("cuda")
    
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=3251, shuffle=True)

    X_test = sys.argv[3]
    Y_test = sys.argv[4]
    X_test = np.loadtxt(X_test, delimiter = ',').astype(np.float32) / 255.0
    Y_test = np.loadtxt(Y_test, delimiter = ',').astype(np.float32).reshape(1, -1)
    Y_test = Y_test.squeeze()
    X_test = torch.tensor(X_test, dtype=torch.float32).to("cuda")
    Y_test = torch.tensor(Y_test, dtype=torch.long).to("cuda")
    test_dataset = TensorDataset(X_test, Y_test)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    epochs = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')