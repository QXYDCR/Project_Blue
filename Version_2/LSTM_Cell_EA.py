

from Utils import *
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------- 超参数设置 ---------------------

# 数据来源文件
good_file = 'data/good.txt'
bad_file  = 'data/bad.txt'
n_class = 2


# --------------------- 获取词典 ---------------------
pos_sentences, neg_sentences, vocab = Prepare_data(good_file, bad_file)


# --------------------- 建立词典模型 ---------------------
# dataset: 每句话中，单词数不等
dataset, labels = text_to_vec_word2vec(pos_sentences, neg_sentences, vocab)

# 划分数据集
# Randomly permute a sequence, or return a permuted range.
indices = np.random.permutation(len(dataset))

# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]

# 生成不同的数据集
train_data, valid_data, test_data = split_dataset(dataset)
train_label, valid_label, test_label = split_dataset(labels)


# --------------------- 建立模型 ---------------------
class LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_Cell, self).__init__()
        # 嵌入层:
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, 20)
        self.lstm_cell = nn.LSTMCell(20, hidden_size)
        self.fc = nn.Linear(hidden_size, n_class)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.p = 0.5

    def forward(self, x, h, c):
        # x: [1] --> [1, hidden]

        x = self.embed(x)
        F.dropout(x, p=self.p, training=True)
        # [
        h, c = self.lstm_cell(x, (h, c))
        # 要尝试将c和x连接在一起

        out = self.fc(c)

        return h, c, out

    def predict(self, c):
        return self.fc(c)

    def initH(self):
        # 注意hidden和cell的维度都是batch_size, hidden_size
        hidden = torch.zeros(1, self.hidden_size)
        # 对隐单元内部的状态cell的初始化，全0
        cell = torch.zeros(1, self.hidden_size)
        return hidden, cell

model = LSTM_Cell(len(vocab), 10)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

h, c = model.initH()
out = torch.randn(1, 2)

losses = []
rights = []

# for episode in range(10):
#
#     for i, data in enumerate(zip(train_data, train_label)):
#         optimizer.zero_grad()
#         x, y = data
#         # [Seq_len, ] --> [Seq, 1]
#         x = torch.LongTensor(x).unsqueeze(1)
#         y = torch.LongTensor(np.array([y]))
#
#         # *** 这句话非常重要，https://blog.csdn.net/qq_31375855/article/details/107568057
#         # 防止在进行梯度BP时，h,c已经被释放了，所以需要初始化
#         # 但还是不是很懂，初始化后，不又得重新训练吗
#         h, c = model.initH()
#
#         for i in range(x.shape[0]):
#             h, c, out = model(x[i], h, c)
#
#         loss = loss_func(out, y)
#         loss.backward()
#         optimizer.step()
#
#     for i, data in enumerate(zip(valid_data, valid_label)):
#         x, y = data
#         # [Seq_len, ] --> [Seq, 1]
#         x = torch.LongTensor(x).unsqueeze(1)
#         y = torch.LongTensor(np.array([y]))
#         for i in range(x.shape[0]):
#             h, c, out = model(x[i], h, c)
#
#         loss = loss_func(out, y)
#         acc = get_accurate(out, y)
#         losses.append(loss)
#         rights.append(acc)
#     right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
#     print('验证数据准确率: {:.5f}, loss: {:.5f}'.format(right_ratio, torch.mean(torch.stack(losses))))
#
#     torch.save(model.state_dict(), 'LSTM_CELL.ztl')
#     print("Save parameter")

model.load_state_dict(torch.load('LSTM_CELL.ztl'))


for i, data in enumerate(zip(test_data, test_label)):
    optimizer.zero_grad()
    x, y = data
    # [Seq_len, ] --> [Seq, 1]
    x = torch.LongTensor(x).unsqueeze(1)
    y = torch.LongTensor(np.array([y]))

    h, c = model.initH()

    for i in range(x.shape[0]):
        h, c, out = model(x[i], h, c)

    loss = loss_func(model.predict(c), y)
    acc = get_accurate(out, y)
    losses.append(loss)
    rights.append(acc)

right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
print('测试数据准确率: {:.5f}, loss: {:.5f}'.format(right_ratio, torch.mean(torch.stack(losses))))






