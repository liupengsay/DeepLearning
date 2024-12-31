import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# 自定义 AUC 损失函数
# 自定义 AUC 损失函数
class AUCLoss(nn.Module):
    def __init__(self):
        super(AUCLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_pred 是模型的预测概率，y_true 是真实标签
        # 我们需要确保 y_pred 是概率值而不是预测的类别
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        # 使用 sklearn 计算 AUC
        auc_score = roc_auc_score(y_true, y_pred)

        # 返回 AUC 的负值作为损失
        # 转换为 PyTorch 张量并返回
        return torch.tensor(1 - auc_score, dtype=torch.float32).cuda()


# 假设文本数据（query 和 doc）是以词的形式表示的
class CTRCVRModel(nn.Module):
    def __init__(self, input_dim, user_id_dim, item_id_dim, term_vocab_size, embedding_dim=8):
        super(CTRCVRModel, self).__init__()

        # 处理数值特征
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        # 处理 ID 特征的嵌入层
        self.user_embedding = nn.Embedding(user_id_dim, embedding_dim)
        self.item_embedding = nn.Embedding(item_id_dim, embedding_dim)

        # 处理 query 和 doc 的词汇嵌入层
        self.term_embedding = nn.Embedding(term_vocab_size, embedding_dim)  # 对每个词进行嵌入

        # 输出层：CTR 和 CVR 的预测
        self.fc_ctr = nn.Linear(32 + 2 * embedding_dim + embedding_dim * 2, 1)  # CTR预测
        self.fc_cvr = nn.Linear(32 + 2 * embedding_dim + embedding_dim * 2, 1)  # CVR预测
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_terms, doc_terms, user_id, item_id, x):
        # 处理数值特征
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 获取 ID 特征的嵌入
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)

        # 获取 query 和 doc 中每个词的嵌入
        query_emb = self.term_embedding(query_terms)  # [batch_size, query_len, embedding_dim]
        doc_emb = self.term_embedding(doc_terms)  # [batch_size, doc_len, embedding_dim]

        # 对 query 和 doc 嵌入求平均
        query_emb_avg = torch.mean(query_emb, dim=1)  # [batch_size, embedding_dim]
        doc_emb_avg = torch.mean(doc_emb, dim=1)  # [batch_size, embedding_dim]

        # 拼接所有特征：数值特征 + 用户和物品嵌入 + query 和 doc 的嵌入
        x = torch.cat([x, user_emb, item_emb, query_emb_avg, doc_emb_avg], dim=-1)

        # 计算 CTR 和 CVR
        ctr = self.sigmoid(self.fc_ctr(x))
        cvr = self.sigmoid(self.fc_cvr(x))

        return ctr, cvr


class CTRCVRDataset(Dataset):
    def __init__(self, query_terms, doc_terms, user_id, item_id, X, y_ctr, y_cvr):
        self.query_terms = query_terms
        self.doc_terms = doc_terms
        self.user_id = user_id
        self.item_id = item_id
        self.X = X
        self.y_ctr = y_ctr
        self.y_cvr = y_cvr

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        query_term = self.query_terms[idx]
        doc_term = self.doc_terms[idx]
        user_id = self.user_id[idx]
        item_id = self.item_id[idx]
        X = self.X[idx]
        y_ctr = self.y_ctr[idx]
        y_cvr = self.y_cvr[idx]

        return query_term, doc_term, user_id, item_id, X, y_ctr, y_cvr


sample_num = 5000
# 示例数据（包括 query 和 doc 的词汇）
query_terms = torch.randint(0, 100, (sample_num, 10))  # 1000个样本，每个样本10个term
doc_terms = torch.randint(0, 100, (sample_num, 50))  # 1000个样本，每个样本50个term
user_id = torch.randint(0, 100, (sample_num,))  # 1000个用户ID
item_id = torch.randint(0, 200, (sample_num,))  # 1000个物品ID
X = torch.rand(sample_num, 3)  # 假设有64个数值型特征
y_ctr = torch.randint(0, 2, (sample_num,))  # CTR标签
y_cvr = torch.randint(0, 2, (sample_num,))  # CVR标签
# 划分训练集和验证集
X_train, X_val, user_id_train, user_id_val, item_id_train, item_id_val, query_terms_train, query_terms_val, doc_terms_train, doc_terms_val, y_ctr_train, y_ctr_val, y_cvr_train, y_cvr_val = train_test_split(
    X, user_id, item_id, query_terms, doc_terms, y_ctr, y_cvr, test_size=0.2, random_state=42)

# 创建模型，损失函数和优化器
term_vocab_size = 100  # 假设词汇大小为100
user_id_dim = len(np.unique(user_id.numpy()))  # 用户ID的总数
item_id_dim = len(np.unique(item_id.numpy()))  # 物品ID的总数
model = CTRCVRModel(X_train.shape[1], user_id_dim, item_id_dim, term_vocab_size).cuda()
auc_loss = AUCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设 query_terms, doc_terms, user_id, item_id, X, y_ctr 和 y_cvr 已经准备好
train_dataset = CTRCVRDataset(query_terms_train, doc_terms_train, user_id_train, item_id_train, X_train, y_ctr_train,
                              y_cvr_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CTRCVRDataset(query_terms_val, doc_terms_val, user_id_val, item_id_val, X_val, y_ctr_val, y_cvr_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 训练阶段
    for batch_idx, (
    query_terms_batch, doc_terms_batch, user_id_batch, item_id_batch, X_batch, y_ctr_batch, y_cvr_batch) in enumerate(
            train_dataloader):
        query_terms_batch = query_terms_batch.cuda()
        doc_terms_batch = doc_terms_batch.cuda()
        user_id_batch = user_id_batch.cuda()
        item_id_batch = item_id_batch.cuda()
        X_batch = X_batch.cuda()
        y_ctr_batch = y_ctr_batch.cuda()
        y_cvr_batch = y_cvr_batch.cuda()

        # 前向传播
        ctr_pred, cvr_pred = model(query_terms_batch, doc_terms_batch, user_id_batch, item_id_batch, X_batch)

        # 计算损失
        loss_ctr = auc_loss(ctr_pred.squeeze(), y_ctr_batch)
        loss_cvr = auc_loss(cvr_pred.squeeze(), y_cvr_batch)
        loss = loss_ctr + loss_cvr

        # 反向传播
        optimizer.zero_grad()
        # 确保梯度计算
        if loss.requires_grad:
            loss.backward()
        else:
            loss = loss.clone().detach().requires_grad_()  # 如果没有requires_grad，确保 loss 会进行梯度计算
            loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每10步打印一次训练损失
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {running_loss / (batch_idx + 1):.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (query_terms_batch, doc_terms_batch, user_id_batch, item_id_batch, X_batch, y_ctr_batch,
                        y_cvr_batch) in enumerate(val_dataloader):
            query_terms_batch = query_terms_batch.cuda()
            doc_terms_batch = doc_terms_batch.cuda()
            user_id_batch = user_id_batch.cuda()
            item_id_batch = item_id_batch.cuda()
            X_batch = X_batch.cuda()
            y_ctr_batch = y_ctr_batch.cuda()
            y_cvr_batch = y_cvr_batch.cuda()

            # 前向传播
            ctr_pred, cvr_pred = model(query_terms_batch, doc_terms_batch, user_id_batch, item_id_batch, X_batch)

            # 计算损失
            loss_ctr = auc_loss(ctr_pred.squeeze(), y_ctr_batch)
            loss_cvr = auc_loss(cvr_pred.squeeze(), y_cvr_batch)
            val_loss += (loss_ctr + loss_cvr).item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_dataloader):.4f}")