import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
assert torch.cuda.is_available()

# 设置全局默认设备为 GPU，如果没有可用的 GPU，则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.device(device)


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义数据预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像或 ndarray 转换为 tensor，并归一化至 [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化至 [-1, 1]
])

# 下载训练集和测试集并进行预处理
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

res = train_set.data[10]

# 将矩阵转换为 PIL Image 对象
image = Image.fromarray(res)
# 保存图片
image.save('output.png')


batch_size = 4
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# 实例化模型
model = CNN()
model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 在训练集上训练模型
for epoch in range(5):  # 多次遍历数据集
    model.train()
    running_loss = 0.0
    i = 0
    for data in train_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        i += 1
    print('Finished Training')
    print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 200))
    running_loss = 0.0
    # 在测试集上验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
