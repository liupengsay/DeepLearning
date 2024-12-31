import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import numpy as np

# 加载预训练的 BERT 模型和 Tokenizer
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 定义一个数据集类
class QueryDocDataset(Dataset):
    def __init__(self, queries, docs, labels, tokenizer, max_len):
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        doc = self.docs[idx]
        label = self.labels[idx]

        # 对 query 和 doc 进行 tokenization
        encoding_query = self.tokenizer(query, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        encoding_doc = self.tokenizer(doc, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        # 获取 tokenized 输入的所有字段
        return {
            'query_input_ids': encoding_query['input_ids'].squeeze(),
            'query_attention_mask': encoding_query['attention_mask'].squeeze(),
            'doc_input_ids': encoding_doc['input_ids'].squeeze(),
            'doc_attention_mask': encoding_doc['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

# 定义一个模型类
class QueryDocModel(nn.Module):
    def __init__(self, bert_model, embedding_dim=128):
        super(QueryDocModel, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(bert_model.config.hidden_size, embedding_dim)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        # 获取 query 和 doc 的嵌入
        query_output = self.bert_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
        doc_output = self.bert_model(input_ids=doc_input_ids, attention_mask=doc_attention_mask)

        # 获取 [CLS] token 对应的向量
        query_embedding = query_output.last_hidden_state[:, 0, :]
        doc_embedding = doc_output.last_hidden_state[:, 0, :]

        # 对嵌入进行处理，生成最后的嵌入
        query_embedding = self.fc(query_embedding)
        doc_embedding = self.fc(doc_embedding)

        return query_embedding, doc_embedding

# 定义损失函数：余弦相似度损失
def cosine_similarity_loss(query_embedding, doc_embedding, label):
    cos_sim = nn.functional.cosine_similarity(query_embedding, doc_embedding)
    loss = nn.MSELoss()(cos_sim, label)
    return loss

# 加载数据
queries = ["今天的天气怎么样？", "如何提升搜索排名？"]
docs = ["今天天气晴，适合出游。", "提高搜索排名的方法有很多。"]
labels = [1.0, 0.0]  # 标签：1.0表示相关，0.0表示不相关

# 数据预处理
max_len = 64
dataset = QueryDocDataset(queries, docs, labels, tokenizer, max_len)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 定义模型和优化器
model = QueryDocModel(bert_model, embedding_dim=128)
optimizer = Adam(model.parameters(), lr=1e-5)

# 训练过程
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()

        # 获取数据
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']
        doc_input_ids = batch['doc_input_ids']
        doc_attention_mask = batch['doc_attention_mask']
        label = batch['label']

        # 获取 query 和 doc 的嵌入
        query_embedding, doc_embedding = model(query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask)

        # 计算损失
        loss = cosine_similarity_loss(query_embedding, doc_embedding, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# 模型训练完毕，可以使用 query 的嵌入进行推理任务
model.eval()  # 设置为评估模式
query = "如何提升搜索排名？"
doc = "提升搜索排名的一些方法包括优化内容、提高用户体验等。"

# 将输入文本 tokenized
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64)
doc_inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=64)

# 获取 query 和 doc 的嵌入
with torch.no_grad():
    query_embedding, doc_embedding = model(
        inputs['input_ids'], inputs['attention_mask'],
        doc_inputs['input_ids'], doc_inputs['attention_mask']
    )

# 计算余弦相似度
cos_sim = nn.functional.cosine_similarity(query_embedding, doc_embedding)
print(f"Query and doc similarity: {cos_sim.item()}")


query = "我的名字叫小芳"
doc = "他不是小芳"

# 将输入文本 tokenized
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=64)
doc_inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=64)

# 获取 query 和 doc 的嵌入
with torch.no_grad():
    query_embedding, doc_embedding = model(
        inputs['input_ids'], inputs['attention_mask'],
        doc_inputs['input_ids'], doc_inputs['attention_mask']
    )

# 计算余弦相似度
cos_sim = nn.functional.cosine_similarity(query_embedding, doc_embedding)
print(f"Query and doc similarity: {cos_sim.item()}")