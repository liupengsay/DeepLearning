import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

assert torch.cuda.is_available()


# 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.device(device)

# 加载 MNIST 训练集和测试集并进行预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST(root='../data/', train=True, transform=transform, download=True)
test_dataset = MNIST(root='../data/', train=False, transform=transform, download=True)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型并将模型移动到 GPU 上
model = Model()
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        # 将训练数据移动到 GPU 上
        images, labels = images.to(device), labels.to(device)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images.view(-1, 28 * 28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}")

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 将测试数据移动到 GPU 上
            images, labels = images.to(device), labels.to(device)

            outputs = model(images.view(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
