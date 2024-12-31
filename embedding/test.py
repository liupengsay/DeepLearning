# import torch
# from transformers import BertTokenizer, BertModel
#
# # 1. 加载预训练模型和分词器
# model_name = 'bert-base-chinese'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
#
# # 2. 输入中文查询
# query = "如何使用BERT生成中文查询的embedding"
#
# # 3. 将查询进行分词并转换为模型输入格式
# inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#
# # 4. 获取模型的输出（包括最后一层的 hidden states）
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # 5. 获取最后一层的隐藏状态（hidden states）
# last_hidden_states = outputs.last_hidden_state
#
# # 6. 计算查询的 embedding（通常使用[CLS] token的表示作为文本表示）
# query_embedding = last_hidden_states[0][0].numpy()  # [CLS] token 对应的是第一个位置
#
# print("查询的embedding:", query_embedding)


import torch
from transformers import BertTokenizer


def token_and_mask_query(query, tokenizer):
    query_t = tokenizer(query, return_tensors="pt", padding=True)
    tokens = tokenizer.tokenize(query)  # 直接用tokenizer进行分词

    term_t = [tokenizer.encode(token, add_special_tokens=False) for token in tokens]  # 获取每个token的ID

    # 初始化mask张量，形状为[ngrams+2, max_query_len]
    mask = torch.zeros(len(tokens) + 2, query_t["input_ids"].shape[1])

    # 填充mask
    for i in range(1, len(tokens) + 1):
        for j in term_t[i - 1]:
            mask[i, query_t["input_ids"][0] == j] = 1

    return query_t, mask


# 示例
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
query = "BERT是一个强大的模型"
query_t, mask = token_and_mask_query(query, tokenizer)

print("Tokenized query:", query_t)
print("Mask:", mask)
