# import torch
# from transformers import BertTokenizer, BertModel
#
# # 加载预训练的 BERT 中文模型和tokenizer
# model_name = "bert-base-chinese"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name, output_attentions=True)
#
# # 输入查询
# query = "今天的天气怎么样？"
#
# # 将文本转换为 BERT 输入格式
# inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#
# # 获取输出，包括注意力权重
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # 输出的last_hidden_state是一个形状为 [batch_size, sequence_length, hidden_size] 的张量
# # 输出的attentions是一个包含每一层注意力矩阵的元组，每一层都是 [batch_size, num_heads, seq_len, seq_len] 形状
# attentions = outputs.attentions
#
# # 打印每一层的注意力矩阵
# for layer_num, attention in enumerate(attentions):
#     print(f"Layer {layer_num + 1} attention shape: {attention.shape}")
#
#     # 获取第一个 batch 和第一个 attention head 的注意力矩阵
#     attention_matrix = attention[0, 0].cpu().numpy()  # 这里我们取第一个 batch 和第一个 attention head
#     print(f"Attention Matrix for Layer {layer_num + 1}:\n", attention_matrix)
#
#     # 获取每个词的 attention 权重，可以选择对每一层的 attention 做加权平均
#     term_weights = attention_matrix.sum(axis=0)  # 对每个词进行加权求和
#     print(f"Term Weights for Layer {layer_num + 1}:\n", term_weights)
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 假设我们已经有了 attention_matrix
# sns.heatmap(attentions, cmap='Blues', xticklabels=query.split(), yticklabels=query.split())
# plt.title("Attention Heatmap")
# plt.show()
#
# import torch
# from transformers import BertTokenizer, BertModel
# import numpy as np
#
# # 加载预训练的 BERT 中文模型和tokenizer
# model_name = "bert-base-chinese"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name, output_attentions=True)
#
# # 输入查询
# query = "今天的天气怎么样？"
#
# # 将文本转换为 BERT 输入格式
# inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#
# # 获取输出，包括注意力权重
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # 输出的 attentions 是一个包含每一层注意力矩阵的元组，每一层都是 [batch_size, num_heads, seq_len, seq_len] 形状
# attentions = outputs.attentions
#
# # 获取第一层的注意力权重
# attention_layer = attentions[0]  # 获取第一层注意力矩阵 (num_heads, seq_len, seq_len)
# attention_matrix = attention_layer[0, 0].cpu().numpy()  # 选择第一个batch和第一个attention head
#
# # 打印注意力矩阵
# print("Attention matrix for the first layer and first head:")
# print(attention_matrix)
#
# # 获取每个分词的权重
# # 对每一层的 attention 权重进行加权求和
# term_weights = attention_matrix.sum(axis=0)  # 对每个词进行加权求和
#
# # 打印每个分词的权重
# tokens = tokenizer.tokenize(query)  # 获取 query 对应的分词
# term_weights = term_weights[:len(tokens)]  # 截取与词数匹配的权重
#
# print("\nTerm Weights for each token:")
# for token, weight in zip(tokens, term_weights):
#     print(f"Token: {token}, Weight: {weight:.4f}")

import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 加载预训练的 BERT 中文模型和tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

# 已分词的 query
query = ["今天", "的", "天气", "怎么样", "？"]

# 将已分词的 query 拼接成一个完整的句子
query_text = " ".join(query)  # 将词用空格连接，形成一个句子

# 将已分词的文本转换为 BERT 输入格式
inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)

# 获取输出，包括注意力权重
with torch.no_grad():
    outputs = model(**inputs)

# 输出的 attentions 是一个包含每一层注意力矩阵的元组，每一层都是 [batch_size, num_heads, seq_len, seq_len] 形状
attentions = outputs.attentions

# 获取第一层的注意力权重
attention_layer = attentions[0]  # 获取第一层注意力矩阵 (num_heads, seq_len, seq_len)
attention_matrix = attention_layer[0, 0].cpu().numpy()  # 选择第一个batch和第一个attention head

# 打印注意力矩阵
print("Attention matrix for the first layer and first head:")
print(attention_matrix)

# 获取每个分词的权重
# 对每一层的 attention 权重进行加权求和
term_weights = attention_matrix.sum(axis=0)  # 对每个词进行加权求和

# 打印每个分词的权重
tokens = tokenizer.tokenize(query_text)  # 获取 BERT 分词后的结果
term_weights = term_weights[:len(tokens)]  # 截取与词数匹配的权重

print("\nTerm Weights for each token:")
for token, weight in zip(tokens, term_weights):
    print(f"Token: {token}, Weight: {weight:.4f}")
