import torch

from Utils import *
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------- 超参数设置 ---------------------

# 数据来源文件
good_file = 'data/Demo_good.txt'
bad_file  = 'data/Demo_bad.txt'
n_class = 2
ALPHA = 0.5

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

# 建立情感词典
pos_dict, neg_dict = get_emotion_dict(train_data, train_label, vocab)


# --------------------- 建立模型 ---------------------
class LSTM_Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_Actor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 嵌入层:
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        # [1,] --> [1, 10]
        x = self.embed(x)
        h, c = self.lstm_cell(x, (h, c))
        out = self.fc(c)

        return h, c, out

    def initH(self):
        # 注意hidden和cell的维度都是batch_size, hidden_size
        hidden = torch.randn(1, self.hidden_size)
        # 对隐单元内部的状态cell的初始化，全0
        cell = torch.randn(1, self.hidden_size)
        return hidden, cell

actor = LSTM_Actor(len(vocab), 10, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)



# --------------------- 建立模型 ---------------------
test_acc = []
test_loss = []
train_acc = []
train_loss = []


# for episode in range(5):
#     for i, data in enumerate(zip(train_data, train_label)):
#         x, y = data
#         x = torch.LongTensor(x).view(-1, 1)
#         y = torch.LongTensor([y])
#
#         h, c = actor.initH()
#         true_actions = []
#         pred_actions = []
#         actions_pair = []
#         for i in range(x.shape[0]):
#
#             h, c, out = actor(x[i], h, c)
#             out = torch.log_softmax(out, 1)
#
#             actions_pair.append(out)
#             if idx2word(x[i], vocab) in pos_dict:
#                 true_actions.append(0)
#             else:
#                 true_actions.append(1)
#
#             pred_actions.append(torch.argmax(out, dim = 1))
#
#         # 回合更新Actor参数
#         true_actions = torch.LongTensor(true_actions)
#         pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
#         # 将含有tensor的list转化为tensor
#         actions_pair = torch.cat(tuple(actions_pair), 0)
#
#         H = F.cross_entropy(actions_pair, true_actions)
#         L_L_ = (true_actions == pred_actions).sum().item() / x.shape[0]
#
#         R = H + ALPHA * L_L_
#         selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
#         loss_actor = -selected_logprobs.mean()
#
#         optimizer.zero_grad()
#         loss_actor.backward()
#         optimizer.step()
#
#         # 添加到列表中
#         train_loss.append(loss_actor)
#         # acc = get_accurate(actions_pair, y)
#         # train_acc.append(acc)
#         bbb = (pred_actions.squeeze() == true_actions)
#         train_acc.append((pred_actions.squeeze() == true_actions).sum() / x.shape[0])
#
#
#     print("episode: {0}, loss: {1}, acc: {2}".format(episode + 1, torch.mean(torch.stack(train_loss)),
#                                                      torch.mean(torch.stack(train_acc))))

#     torch.save(actor.state_dict(), 'LSTM_ACTOR.ztl')
#     print("Save parameters")


# 测试数据
actor.load_state_dict(torch.load('LSTM_ACTOR.ztl'))

for i, data in enumerate(zip(test_data, test_label)):
    x, y = data
    x = torch.LongTensor(x).view(-1, 1)
    y = torch.LongTensor([y])

    h, c = actor.initH()
    true_actions = []
    pred_actions = []
    actions_pair = []
    for i in range(x.shape[0]):
        h, c, out = actor(x[i], h, c)
        out = torch.log_softmax(out, 1)

        actions_pair.append(out)
        if idx2word(x[i], vocab) in pos_dict:
            true_actions.append(0)
        else:
            true_actions.append(1)

        pred_actions.append(torch.argmax(out, dim=1))

    # 回合更新Actor参数
    true_actions = torch.LongTensor(true_actions)
    pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
    # 将含有tensor的list转化为tensor
    actions_pair = torch.cat(tuple(actions_pair), 0)

    H = F.cross_entropy(actions_pair, true_actions)
    L_L_ = (true_actions == pred_actions).sum().item() / x.shape[0]

    R = H + ALPHA * L_L_
    selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
    loss_actor = -selected_logprobs.mean()

    # 添加到列表中
    train_loss.append(loss_actor)
    # acc = get_accurate(actions_pair, y)
    # train_acc.append(acc)
    bbb = (pred_actions.squeeze() == true_actions)
    train_acc.append((pred_actions.squeeze() == true_actions).sum() / x.shape[0])

print("episode: {0}, loss: {1}, acc: {2}".format(1, torch.mean(torch.stack(train_loss)),
                                                 torch.mean(torch.stack(train_acc))))





































