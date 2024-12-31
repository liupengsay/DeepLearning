import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split


# 自定义 AUC 损失函数
class AUCLoss(nn.Module):
    def __init__(self):
        super(AUCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_pred 是模型的预测概率，y_true 是真实标签
        # 我们需要确保 y_pred 是概率值而不是预测的类别
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        # 使用 sklearn 计算 AUC
        auc_score = roc_auc_score(y_true, y_pred)

        # 返回 AUC 的负值作为损失
        # 转换为 PyTorch 张量并返回
        return torch.tensor(1 - auc_score, dtype=torch.float32).cuda()


# 示例模型，增加了 ID 特征的处理
class CTRCVRModel(nn.Module):
    def __init__(self, input_dim, user_id_dim, item_id_dim, embedding_dim=8):
        super(CTRCVRModel, self).__init__()

        # 处理数值特征
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        # 处理 ID 特征的嵌入层
        self.user_embedding = nn.Embedding(user_id_dim, embedding_dim)  # user_id 的嵌入
        self.item_embedding = nn.Embedding(item_id_dim, embedding_dim)  # item_id 的嵌入

        # 输出层：CTR 和 CVR 的预测
        self.fc_ctr = nn.Linear(32 + 2 * embedding_dim, 1)  # CTR预测，嵌入层的输出拼接到最终预测
        self.fc_cvr = nn.Linear(32 + 2 * embedding_dim, 1)  # CVR预测，嵌入层的输出拼接到最终预测

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, user_id, item_id):
        # 通过数值特征进行计算
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 获取 ID 特征的嵌入
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        # 拼接数值特征和 ID 嵌入特征
        x = torch.cat([x, user_emb, item_emb], dim=-1)

        # 计算 CTR 和 CVR
        ctr = self.sigmoid(self.fc_ctr(x))
        cvr = self.sigmoid(self.fc_cvr(x))

        return ctr, cvr


# 模拟数据（包括 ID 特征）
X = np.random.rand(5000, 3)  # 假设有3个数值型特征
user_id = np.random.randint(0, 100, 5000)  # 用户ID（类别特征）
item_id = np.random.randint(0, 200, 5000)  # 物品ID（类别特征）
y_ctr = np.random.randint(0, 2, 5000)  # CTR标签
y_cvr = np.random.randint(0, 2, 5000)  # CVR标签

# 划分训练集和验证集
X_train, X_val, user_id_train, user_id_val, item_id_train, item_id_val, y_ctr_train, y_ctr_val, y_cvr_train, y_cvr_val = train_test_split(
    X, user_id, item_id, y_ctr, y_cvr, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).cuda()
user_id_train_tensor = torch.tensor(user_id_train, dtype=torch.long).cuda()  # 用户 ID 需要是 long 类型
item_id_train_tensor = torch.tensor(item_id_train, dtype=torch.long).cuda()  # 物品 ID 需要是 long 类型
y_ctr_train_tensor = torch.tensor(y_ctr_train, dtype=torch.float32).cuda()
y_cvr_train_tensor = torch.tensor(y_cvr_train, dtype=torch.float32).cuda()
user_id_val_tensor = torch.tensor(user_id_val, dtype=torch.long).cuda()
item_id_val_tensor = torch.tensor(item_id_val, dtype=torch.long).cuda()
y_ctr_val_tensor = torch.tensor(y_ctr_val, dtype=torch.float32).cuda()
y_cvr_val_tensor = torch.tensor(y_cvr_val, dtype=torch.float32).cuda()

# 创建模型，损失函数和优化器
input_dim = X_train.shape[1]
user_id_dim = len(np.unique(user_id))  # 用户 ID 的总数
item_id_dim = len(np.unique(item_id))  # 物品 ID 的总数
model = CTRCVRModel(input_dim, user_id_dim, item_id_dim).cuda()
auc_loss = AUCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    ctr_pred, cvr_pred = model(X_train_tensor, user_id_train_tensor, item_id_train_tensor)

    # 计算损失
    loss_ctr = auc_loss(ctr_pred.squeeze(), y_ctr_train_tensor)
    loss_cvr = auc_loss(cvr_pred.squeeze(), y_cvr_train_tensor)
    loss = loss_ctr + loss_cvr

    # 确保梯度计算
    if loss.requires_grad:
        loss.backward()
    else:
        loss = loss.clone().detach().requires_grad_()  # 如果没有requires_grad，确保 loss 会进行梯度计算
        loss.backward()
    optimizer.step()

    # 输出训练结果
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 验证
    model.eval()
    with torch.no_grad():
        ctr_pred_val, cvr_pred_val = model(X_val_tensor, user_id_val_tensor, item_id_val_tensor)
        val_loss_ctr = auc_loss(ctr_pred_val.squeeze(), y_ctr_val_tensor)
        val_loss_cvr = auc_loss(cvr_pred_val.squeeze(), y_cvr_val_tensor)
        val_loss = val_loss_ctr + val_loss_cvr
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}')
