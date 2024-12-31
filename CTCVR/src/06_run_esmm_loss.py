import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# 生成一些模拟数据
def generate_sample_data(num_samples=10000):
    # 生成模拟query, doc, 和特征数据
    query_ids = np.random.randint(1, 1000, size=num_samples)
    doc_ids = np.random.randint(1, 500, size=num_samples)
    utdid = np.random.randint(1, 10000, size=num_samples)

    # CTR (Click-through rate): 随机生成 0 或 1 (点击与否)
    click = np.random.randint(0, 2, size=num_samples)

    # CVR (Conversion rate): 转化率, 点击后是否转化
    convert = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])  # 假设70%点击后不转化，30%转化

    # 生成一些随机数值型特征，如展示位置，用户设备类型等
    position = np.random.randint(1, 10, size=num_samples)  # 假设位置为1到10之间
    device_type = np.random.choice([0, 1], size=num_samples)  # 假设0为移动设备，1为桌面设备
    doc_length = np.random.randint(50, 500, size=num_samples)  # 假设文档长度

    # 创建DataFrame
    data = pd.DataFrame({
        'query_id': query_ids,
        'doc_id': doc_ids,
        'utdid': utdid,
        'click': click,
        'convert': convert,
        'position': position,
        'device_type': device_type,
        'doc_length': doc_length
    })

    return data


# 生成数据样例
data = generate_sample_data(100000)

# 数据划分
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data.head())


class ESMM(nn.Module):
    def __init__(self, feature_size, hidden_dim=64):
        super(ESMM, self).__init__()

        # 特征嵌入层
        self.query_embedding = nn.Embedding(1000, 8)  # 假设query_id有1000种
        self.doc_embedding = nn.Embedding(500, 8)  # 假设doc_id有500种
        self.utdid_embedding = nn.Embedding(10000, 8)  # 假设utdid有10000种

        # 全连接层
        self.fc_ctr = nn.Sequential(
            nn.Linear(8 + 8 + 8 + 3, hidden_dim),  # 嵌入维度 + 数值特征（position, device_type, doc_length）
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0到1之间的概率值
        )

        self.fc_cvr = nn.Sequential(
            nn.Linear(8 + 8 + 8 + 3, hidden_dim),  # 相同的输入维度
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0到1之间的概率值
        )

    def forward(self, query_id, doc_id, utdid, position, device_type, doc_length):
        # 嵌入查询、文档和用户特征
        query_emb = self.query_embedding(query_id)
        doc_emb = self.doc_embedding(doc_id)
        utdid_emb = self.utdid_embedding(utdid)

        # 合并所有特征
        x = torch.cat(
            [query_emb, doc_emb, utdid_emb, position.unsqueeze(1), device_type.unsqueeze(1), doc_length.unsqueeze(1)],
            dim=1)

        # CTR和CVR分别计算
        ctr = self.fc_ctr(x)
        cvr = self.fc_cvr(x)

        return ctr, cvr


# 定义损失函数
class ESMMLoss(nn.Module):
    def __init__(self):
        super(ESMMLoss, self).__init__()

    def forward(self, ctr_pred, cvr_pred, click, convert):
        # CTR损失
        ctr_loss = nn.BCELoss()(ctr_pred.squeeze(), click.float())

        # CVR损失
        cvr_loss = nn.BCELoss()(cvr_pred.squeeze(), convert.float())

        # 总损失
        return ctr_loss + cvr_loss


# 初始化模型和损失函数
model = ESMM(feature_size=128)
criterion = ESMMLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟批处理数据
batch_data = torch.tensor(train_data[['query_id', 'doc_id', 'utdid', 'position', 'device_type', 'doc_length']].values)
batch_labels = torch.tensor(train_data[['click', 'convert']].values)

query_id = batch_data[:, 0]
doc_id = batch_data[:, 1]
utdid = batch_data[:, 2]
position = batch_data[:, 3]
device_type = batch_data[:, 4]
doc_length = batch_data[:, 5]

click = batch_labels[:, 0]
convert = batch_labels[:, 1]

# 训练步骤
model.train()
optimizer.zero_grad()
ctr_pred, cvr_pred = model(query_id, doc_id, utdid, position, device_type, doc_length)
loss = criterion(ctr_pred, cvr_pred, click, convert)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
