import numpy as np
import pandas as pd
import random
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)


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
        return 1 - auc_score

# 定义生成 CTR 和 CVR 数据的函数
def generate_ctr_cvr_data(num_queries=100, num_docs_per_query=50, max_query_length=5, max_doc_length=50):
    queries = []
    docs = []
    ctr_values = []
    cvr_values = []
    query_lengths = []
    doc_lengths = []
    query_doc_similarities = []

    for query_id in range(num_queries):
        # 随机生成一个query，长度在 [1, max_query_length] 之间
        query_length = random.randint(1, max_query_length)
        query = " ".join([f"term{random.randint(1, 100)}" for _ in range(query_length)])

        # 记录查询长度


        # 对每个query生成50个doc
        for doc_id in range(num_docs_per_query):
            # 随机生成一个doc，长度在 [5, max_doc_length] 之间
            doc_length = random.randint(5, max_doc_length)
            doc = " ".join([f"term{random.randint(1, 100)}" for _ in range(doc_length)])

            # 记录文档长度
            doc_lengths.append(doc_length)

            # 计算 CTR (点击率) 和 CVR (转化率)
            # 假设广告内容与用户的兴趣匹配度与query-doc之间的相似度有关
            similarity = np.random.uniform(0, 1)  # 随机生成一个相似度值
            ctr = 1 if similarity > 0.7 else 0  # 如果相似度大于0.7则点击，否则不点击

            # 假设CVR与CTR相关，如果点击（CTR=1），则有一定概率转化
            cvr = 1 if ctr == 1 and random.random() > 0.5 else 0  # 转化率概率设为50%

            # 计算 Query 和 Doc 之间的相似度（假设是基于词的重合度）
            query_terms = set(query.split())
            doc_terms = set(doc.split())
            intersection = len(query_terms.intersection(doc_terms))
            union = len(query_terms.union(doc_terms))
            similarity = intersection / union if union > 0 else 0
            query_doc_similarities.append(similarity)

            queries.append(query)
            docs.append(doc)
            ctr_values.append(ctr)
            cvr_values.append(cvr)
            query_lengths.append(query_length)

    # 返回生成的 DataFrame
    return pd.DataFrame({
        'query': queries,
        'doc': docs,
        'ctr': ctr_values,
        'cvr': cvr_values,
        'query_length': query_lengths,
        'doc_length': doc_lengths,
        'query_doc_similarity': query_doc_similarities
    })

# 生成100个query，每个query下有50个doc，生成5000个数据
data = generate_ctr_cvr_data(num_queries=100, num_docs_per_query=50)

# 查看数据的前几行
print(data.head())

# 准备训练数据
X = data[['query_length', 'doc_length', 'query_doc_similarity']]
y_ctr = data['ctr']
y_cvr = data['cvr']

# 将数据划分为训练集和验证集
X_train, X_val, y_ctr_train, y_ctr_val, y_cvr_train, y_cvr_val = train_test_split(X, y_ctr, y_cvr, test_size=0.2, random_state=42)

# 定义一个简单的联合模型：同时预测 CTR 和 CVR
class CTRCVRModel(nn.Module):
    def __init__(self, input_dim):
        super(CTRCVRModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_ctr = nn.Linear(32, 1)  # CTR预测
        self.fc_cvr = nn.Linear(32, 1)  # CVR预测
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        ctr = self.sigmoid(self.fc_ctr(x))
        cvr = self.sigmoid(self.fc_cvr(x))
        return ctr, cvr

# 定义模型、损失函数和优化器
model = CTRCVRModel(input_dim=X_train.shape[1]).cuda()
criterion_ctr = nn.BCELoss()
criterion_cvr = nn.BCELoss()
#
# criterion_ctr = AUCLoss()
# criterion_cvr = AUCLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换数据为torch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).cuda()
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).cuda()
y_ctr_train_tensor = torch.tensor(y_ctr_train.values, dtype=torch.float32).cuda()
y_cvr_train_tensor = torch.tensor(y_cvr_train.values, dtype=torch.float32).cuda()
y_ctr_val_tensor = torch.tensor(y_ctr_val.values, dtype=torch.float32).cuda()
y_cvr_val_tensor = torch.tensor(y_cvr_val.values, dtype=torch.float32).cuda()

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    ctr_pred, cvr_pred = model(X_train_tensor)

    # 计算损失
    loss_ctr = criterion_ctr(ctr_pred.squeeze(), y_ctr_train_tensor)
    loss_cvr = criterion_cvr(cvr_pred.squeeze(), y_cvr_train_tensor)
    loss = loss_ctr + loss_cvr

    # 反向传播
    loss.backward()
    optimizer.step()

    # 输出训练结果
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 验证
    model.eval()
    with torch.no_grad():
        ctr_pred_val, cvr_pred_val = model(X_val_tensor)
        loss_ctr_val = criterion_ctr(ctr_pred_val.squeeze(), y_ctr_val_tensor)
        loss_cvr_val = criterion_cvr(cvr_pred_val.squeeze(), y_cvr_val_tensor)
        val_loss = loss_ctr_val + loss_cvr_val
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}')
