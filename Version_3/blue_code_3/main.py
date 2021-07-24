from Utils import *

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/Project_Blue/Version_2/data/good.txt'
bad_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/Project_Blue/Version_2/data/bad.txt'


# ------------------------ 2、读取数据 ------------------------
# 读取数据
pos_sentences, neg_sentences,  vocab= read_data_from_pos_neg(good_file, bad_file)


# 加载pos, neg词典
pos_dict_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/' \
                'Project_Blue/Version_3/Emotion_Dict/dict/positive.txt'
neg_dict_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/' \
                'Project_Blue/Version_3/Emotion_Dict/dict/negative.txt'

pos_dict_std = load_dict(pos_dict_file)
neg_dict_std = load_dict(neg_dict_file)


# 加载预定义词向量，并且将自定义词典中的单词映射成词向量
# 预加载的词向量为50维的wiki
word2vec_path = 'C:\\Users\\Chihiro\\Desktop\\RL Paper' \
                '\\Project\\Project_Blue_1\\词向量\\wiki_word2vec_50.bin'
word2vec = build_word2vec(word2vec_path, vocab)
mark_star()


# ------------------------ 3、数值化文本 ------------------------
pos_dataset, pos_labels = convert_emotion_text_to_num(good_file, vocab, lnum=0)
neg_dataset, neg_labels = convert_emotion_text_to_num(bad_file, vocab, lnum=1)

# 混合训练集和验证集
dataset = np.vstack([pos_dataset, neg_dataset])
labels = np.concatenate([pos_labels, neg_labels])

# 划分数据集
# Randomly permute a sequence, or return a permuted range.
indices = np.random.permutation(len(dataset))

# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]

# 对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
# // 表示 int(a / b)
test_size = len(dataset) // 10
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]

# 转载数据集
# train_dataset = TensorDataset(torch.from_numpy(pos_dataset), torch.LongTensor(pos_labels))
# train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True, drop_last = True)
#


# ------------------------ 4、定义网络 ------------------------
class LSTM_Cell_Model(nn.Module):
    def __init__(self, vocab_size = len(vocab), embedding_dim=50, hidden_size=50,
                pretrained_embed = word2vec, is_updata_w2c = True, n_class = 3):
        super(LSTM_Cell_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pretrained_embed = pretrained_embed
        self.is_updata_w2c = is_updata_w2c
        self.n_class = n_class

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = is_updata_w2c

        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, x, h, c):
        # x: [32, ] --> [32, 50]
        x = self.embedding(x)
        # h, c: [32, 50]
        h, c = self.lstm_cell(x, (h, c))

        # [32, 50] --> [32, 2]
        outputs = self.linear(c)
        return h, c, outputs

    def init_H_C(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(1, self.hidden_size)
        c = torch.zeros(1, self.hidden_size)
        return h, c

model = LSTM_Cell_Model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)



# ------------------------ 5、训练网络 ------------------------
save_path = 'JD_Buy.ztl'
max_acc = 0
output = 0

out = 0
for epoch in range(4):
    losses = []
    rights = []
    for idx, data in enumerate(zip(train_data, train_label)):
        x, y = data
        x = torch.LongTensor(x).view(-1, 1)
        y = torch.LongTensor(y)
        # h,c [1, 50]
        h, c = model.init_H_C()

        true_actions = []       # 真实动作
        pred_actions = []       # 预测动作
        actions_pair = []       # 回合预测动作对
        for i in range(x.shape[0]):
            h, c, output = model(x[i], h, c)

            out = torch.log_softmax(output, 1)
            actions_pair.append(out)

            if idx2word(x[i], vocab) in pos_dict_std:
                true_actions.append(0)
            elif idx2word(x[i], vocab) in neg_dict_std:
                true_actions.append(1)
            else:
                true_actions.append(2)

            pred_actions.append(torch.argmax(out, dim = 1))

        # 回合更新Actor参数
        # [50, ]
        true_actions = torch.LongTensor(true_actions)
        # [50, 1]
        pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
        # 将含有tensor的list转化为tensor, [50, 3]
        actions_pair = torch.cat(tuple(actions_pair), 0)

        H = F.cross_entropy(actions_pair, true_actions)
        L_L_ = (true_actions == pred_actions.squeeze()).sum().item() / x.shape[0]

        R = H + 0.5 * L_L_
        selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
        loss_actor = -selected_logprobs.mean()

        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = get_accurate(out, y)
        losses.append(loss)
        rights.append(acc)

        if idx % 200 == 0:
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('验证数据准确率: {:.5f}, loss: {:.5f}'.format(right_ratio, torch.mean(torch.stack(losses))))

            torch.save(model.state_dict(), save_path)
            print("Save parameters")

mark_star()

# # 加载模型
model.load_state_dict(torch.load(save_path))

losses = []
rights = []
model.eval()
out = 0
for i, data in enumerate(zip(test_data, test_label)):

    x, y = data
    # print(convert_num_to_txt(x, vocab))
    x = torch.LongTensor(x).view(-1, 1)
    y = torch.LongTensor(y)

    h, c = model.init_H_C()

    for i in range(x.shape[0]):
        h, c, out = model(x[i], h, c)

    out = torch.log_softmax(out, 1)
    a = torch.argmax(out, 1)
    # print("p: ", a, "T: ", y.item())

    loss = loss_func(out, y)

    acc = get_accurate(out, y)
    losses.append(loss)
    rights.append(acc)

right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
print('测试数据准确率: {:.5f}, loss: {:.5f}'.format(right_ratio, torch.mean(torch.stack(losses))))





