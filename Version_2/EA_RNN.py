import numpy as np
import torch.nn

from Utils import *

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------- 超参数设置 ---------------------

# 数据来源文件
good_file = 'data/Demo_good.txt'
bad_file  = 'data/Demo_bad.txt'


# --------------------- 获取词典 ---------------------
pos_sentences, neg_sentences, vocab = Prepare_data(good_file, bad_file)

# --------------------- 建立词典模型 ---------------------
# 重新准备数据，输入给RNN
# 与词袋模型不同的是。每一个句子在词袋模型中都被表示为了固定长度的向量，其中长度为字典的尺寸
# 在RNN中，每一个句子就是被单独当成词语的序列来处理的，因此序列的长度是与句子等长的

dataset = []
labels = []
sentences = []
# 正例集合
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in vocab:
            # 注意将每个词编码
            new_sentence.append(word2idx(l, vocab))
    #每一个句子都是一个不等长的整数序列
    dataset.append(new_sentence)
    labels.append(0)

# 反例集合
for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in vocab:
            new_sentence.append(word2idx(l, vocab))
    dataset.append(new_sentence)
    labels.append(1)

# [13, 14, 15, 16, 17, 18, 19] ['不错', '穿着', '舒服', '还', '不贵', '挺', '好']
# print(dataset[1], idx_to_sentence(dataset[1], vocab))

# 生成不同的数据集
train_data, valid_data, test_data = split_dataset(dataset)
train_label, valid_label, test_label = split_dataset(labels)


# 一个手动实现的RNN模型

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # LSTM的构造如下：一个embedding层，将输入的任意一个单词映射为一个向量
        # 一个LSTM隐含层，共有hidden_size个LSTM神经元
        # 一个全链接层，外接一个softmax输出
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, 2)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden=None):
        # input尺寸: seq_length
        # 词向量嵌入
        embedded = self.embedding(input)
        # embedded尺寸: seq_length, hidden_size

        # PyTorch设计的LSTM层有一个特别别扭的地方是，输入张量的第一个维度需要是时间步，
        # 第二个维度才是batch_size，所以需要对embedded变形
        embedded = embedded.view(input.data.size()[0], 1, self.hidden_size)
        # embedded尺寸: seq_length, batch_size = 1, hidden_size

        # 调用PyTorch自带的LSTM层函数，注意有两个输入，一个是输入层的输入，另一个是隐含层自身的输入
        # 输出output是所有步的隐含神经元的输出结果，hidden是隐含层在最后一个时间步的状态。
        # 注意hidden是一个tuple，包含了最后时间步的隐含层神经元的输出，以及每一个隐含层神经元的cell的状态

        output, hidden = self.lstm(embedded, hidden)
        # output尺寸: seq_length, batch_size = 1, hidden_size
        # hidden尺寸: 二元组(n_layer = 1 * batch_size = 1 * hidden_size, n_layer = 1 * batch_size = 1 * hidden_size)

        # 我们要把最后一个时间步的隐含神经元输出结果拿出来，送给全连接层
        output = output[-1, ...]
        # output尺寸: batch_size = 1, hidden_size

        # 全链接层
        out = self.fc(output)
        # out尺寸: batch_size = 1, output_size
        # softmax
        out = self.logsoftmax(out)
        return out

    def initHidden(self):
        # 对隐单元的初始化

        # 对隐单元输出的初始化，全0.
        # 注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        # 对隐单元内部的状态cell的初始化，全0
        cell = torch.zeros(self.n_layers, 1, self.hidden_size)
        return (hidden, cell)


class LSTM_Cell(nn.Module):
    def __init__(self, input, hidden=None):
        super(LSTM_Cell, self).__init__()

    def forward(self, x, h, c):

        return x

# 开始训练LSTM网络

# 构造一个LSTM网络的实例
lstm = LSTMNetwork(len(vocab), 10, 2)

# 定义损失函数
cost = torch.nn.NLLLoss()

# 定义优化器
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
records = []

# 开始训练，一共5个epoch，否则容易过拟合
losses = []
rights = []

for epoch in range(3 ):
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data
        x = torch.LongTensor(x).unsqueeze(1)
        # x尺寸：seq_length，序列的长度
        y = torch.LongTensor([y])
        # y尺寸：batch_size = 1, 1
        optimizer.zero_grad()

        # 初始化LSTM隐含层单元的状态
        hidden = lstm.initHidden()
        # hidden: 二元组(n_layer = 1 * batch_size = 1 * hidden_size, n_layer = 1 * batch_size = 1 * hidden_size)

        # 让LSTM开始做运算，注意，不需要手工编写对时间步的循环，而是直接交给PyTorch的LSTM层。
        # 它自动会根据数据的维度计算若干时间步
        output = lstm(x, hidden)
        # output尺寸: batch_size = 1, output_size

        # 损失函数
        loss = cost(output, y)
        losses.append(loss.data.numpy())

        # 反向传播
        loss.backward()
        optimizer.step()

    # 保存参数
    torch.save(lstm.state_dict(), 'LSTM.ztl')
    print("Sava parameters!!!")


for j, val in enumerate(zip(test_data, test_label)):
    x, y = val
    x = torch.LongTensor(x).unsqueeze(1)
    y = torch.LongTensor(np.array([y]))
    hidden = lstm.initHidden()
    output = lstm(x, hidden)
    # 计算校验数据集上的分类准确度
    right = get_accurate(output, y)
    if right[0].numpy() == 0:
        print(idx_to_sentence(x, vocab))
        print(output, y)
    rights.append(right)

right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
print(' 测试准确率: {:.5f}'.format(right_ratio))

#在测试集上计算总的正确率
# vals = [] #记录准确率所用列表
# rights = []
#
# pos = []
# neg = []
# a = '这应该是我在网上买裤子最喜欢的一条的，太赞'
# pos.append(a)
# a = '不错，穿着舒服，还不贵，挺好'
# pos.append(a)
# a = '掌柜人不错，质量还行，服务很算不错的。'
# pos.append(a)
# a = '还不错的老板，下次还来买别的'
# pos.append(a)
#
# a = '宝贝质量不错，很喜欢了。谢谢掌柜。'
# neg.append(a)
#
# data = []
# label = []
# # 正例集合
# for sentence in pos:
#     new_sentence = []
#     for l in sentence:
#         if l in vocab:
#             # 注意将每个词编码
#             new_sentence.append(word2idx(l, vocab))
#     #每一个句子都是一个不等长的整数序列
#     data.append(new_sentence)
#     label.append(0)
#
# # 反例集合
# for sentence in neg:
#     new_sentence = []
#     for l in sentence:
#         if l in vocab:
#             new_sentence.append(word2idx(l, vocab))
#     data.append(new_sentence)
#     label.append(1)
#
# print(data, len(data[0]))
# print(label)






# 绘制误差曲线
# a = [i[0] for i in records]
# b = [i[1] for i in records]
# c = [i[2] for i in records]
# plt.plot(a, label = 'Train Loss')
# plt.plot(b, label = 'Valid Loss')
# plt.plot(c, label = 'Valid Accuracy')
# plt.xlabel('Steps')
# plt.ylabel('Loss & Accuracy')
# plt.legend()
# plt.show()




