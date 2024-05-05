import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Example tabular data (features and labels)
features = np.random.rand(100, 10)  # Example: 100 samples, 10 features
labels = np.random.randint(0, 2, size=(100,))  # Binary classification labels

# Create datasets and dataloaders
train_dataset = CustomDataset(features[:80], labels[:80])
valid_dataset = CustomDataset(features[80:], labels[80:])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)


# Define a simple neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# Initialize the model, loss function, and optimizer
model = Model()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            valid_loss += criterion(outputs.squeeze(), labels).item()
        valid_loss /= len(valid_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}')
