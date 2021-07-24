import numpy as np
import torch

from Utils import *
torch.manual_seed(1234)


# --------------------- 超参数设置 ---------------------

# 数据来源文件
good_file = 'data/good.txt'
bad_file  = 'data/bad.txt'

pos_sentences = []      # 正向切分后的语句
neg_sentences = []      # 负向切分后的语句
vocab = []              # 词典 word2idx

# --------------------- 获取词典 ---------------------
pos_sentences, neg_sentences, vocab = Prepare_data(good_file, bad_file)


# --------------------- 建立词典模型 ---------------------
# dataset: 每句话中，单词数不等
dataset, labels = text_to_vec(pos_sentences, neg_sentences, vocab)


# Randomly permute a sequence, or return a permuted range.
indices = np.random.permutation(len(dataset))

# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]

# 生成不同的数据集
train_data, valid_data, test_data = split_dataset(dataset)
train_label, valid_label, test_label = split_dataset(labels)

# print(idx_to_sentence(train_data[0], vocab), train_label[0])
mark_star()
a = train_data[0]
print(len(a))

# print(idx_to_sentence(train_data[10], vocab), train_label[10])
# print(idx_to_sentence(train_data[2], vocab), train_label[2])
# print(idx_to_sentence(train_data[100], vocab), train_label[100])

# --------------------- 定义模型 ---------------------
model = nn.Sequential(
    nn.Linear(len(vocab), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
)


# --------------------- 训练模型 ---------------------
# 损失函数为交叉熵
criterion = torch.nn.CrossEntropyLoss()
# 优化算法为Adam，可以自动调节学习率
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
records = []


#循环10个Epoch
losses = []
# for epoch in range(5):
#     for i, data in enumerate(zip(train_data, train_label)):
#         x, y = data
#         # 需要将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，也即第一个为1的维度
#         x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
#         # x的尺寸：batch_size=1, len_dictionary
#         # 标签也要加一层外衣以变成1*1的张量
#         y = torch.tensor(np.array([y]), dtype=torch.long)
#
#         # 清空梯度
#         optimizer.zero_grad()
#         # 模型预测
#         predict = model(x)
#         # 计算损失函数
#         loss = criterion(predict, y)
#         # 将损失函数数值加入到列表中
#         losses.append(loss.data.numpy())
#         # 开始进行梯度反传
#         loss.backward()
#         # 开始对参数进行一步优化
#         optimizer.step()
#
#     val_losses = []
#     rights = []
#     # 在所有校验数据集上实验
#     for j, val in enumerate(zip(valid_data, valid_label)):
#         x, y = val
#         x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
#         y = torch.tensor(np.array([y]), dtype=torch.long)
#         predict = model(x)
#         # 调用rightness函数计算准确度
#         right = get_accurate(predict, y, True)
#         rights.append(right)
#         loss = criterion(predict, y)
#         val_losses.append(loss.data.numpy())
#
#     # 将校验集合上面的平均准确度计算出来
#     right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
#     print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
#                                                                 np.mean(val_losses), right_ratio))
#
#     torch.save(model.state_dict(), 'FC.ztl')
#     print("Save parameters")



# --------------------- 测试模型 ---------------------
model.load_state_dict(torch.load('FC.ztl'))


acc = []
aaaaa = []
for i in range(5):
    for i, data in enumerate(zip(test_data, test_label)):
        x, y = data
        x = torch.FloatTensor(x).view(1, -1)
        y = torch.LongTensor(np.array([y]))

        predict = model(x)
        loss = criterion(predict, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accurate = get_accurate(predict, y)
        acc.append(accurate)

    right_ratio = 1.0 * np.sum([i[0] for i in acc]) / np.sum([i[1] for i in acc])
    aaaaa.append(right_ratio)

import matplotlib.pyplot as plt
plt.plot(range(len(aaaaa)), aaaaa)
plt.show()

