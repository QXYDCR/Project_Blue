from Utils import *

"""
        Orange Version 1
Note:
LSTM_Cell 单模块测试
很奇怪
在进行单一LSTM_CELL测试时，在第五轮精度增加了30%，前四轮为62%左右，
最后一轮为90左右

当将最大句子长度固定位30时，效果立即好了起来，到第二轮的时候就已经达到85%以上了
最后训练精度93%

"""



# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'



# ------------------------ 2、读取数据 ------------------------
# 读取数据
pos_sentences, neg_sentences,  vocab = read_data_from_pos_neg(good_file, bad_file)


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
pos_dataset, pos_labels = convert_emotion_text_to_num(good_file, vocab, lnum=0, max_len=30)
neg_dataset, neg_labels = convert_emotion_text_to_num(bad_file, vocab, lnum=1, max_len=30)


# ------------------------ 4、建立数据集 ------------------------
# 将正向、负向集合融合，在随机打乱
dataset, labels = mix_dataset(pos_dataset, pos_labels, neg_dataset, neg_labels)
dataset, labels = random_data(dataset, labels)

# 切分数据集
test_size = len(dataset) // 10
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)
train_dataset = TensorDataset(torch.from_numpy(train_data), torch.LongTensor(train_label))
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True, drop_last = True)


valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]


# ------------------------ 5、定义网络模型 ------------------------

class Actor(nn.Module):
    def __init__(self, vocab_size=len(vocab), embedding_dim=50, hidden_size=50,
                 pretrained_embed=word2vec, is_updata_w2c=True, n_class=2):
        super(Actor, self).__init__()
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
        # x: [1, ] --> [1, 50]
        x = self.embedding(x)
        # h, c: [1, 100]
        h, c = self.lstm_cell(x, (h, c))
        out = self.linear(c)
        return h, c, out

    def init_H_C(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(1, self.hidden_size)
        c = torch.zeros(1, self.hidden_size)
        return h, c

actor = Actor()
opti_actor = torch.optim.Adam(actor.parameters(), lr = 0.001)


class Critic(nn.Module):
    def __init__(self, vocab_size=len(vocab), embedding_dim=50, hidden_size=50,
                 pretrained_embed=word2vec, is_updata_w2c=True, n_class=2, batch = 32):
        super(Critic, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pretrained_embed = pretrained_embed
        self.is_updata_w2c = is_updata_w2c
        self.n_class = n_class
        self.batch = batch

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = is_updata_w2c

        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, x, h, c):
        # x: [1, ] --> [1, 50]
        x = self.embedding(x)
        # h, c: [1, 100]
        h, c = self.lstm_cell(x, (h, c))
        out = self.linear(c)
        return h, c, out

    def init_H_C(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(self.batch, self.hidden_size)
        c = torch.zeros(self.batch, self.hidden_size)
        return h, c

critic = Critic()
loss_critic = nn.CrossEntropyLoss()
opti_critic = torch.optim.Adam(critic.parameters(), lr = 0.001)

class FC_Classifier(nn.Module):
    def __init__(self, hidden_size = 50, n_class = 2):
        super(FC_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_class = n_class

        self.fc = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        x = self.fc(x)
        return x

# FC = FC_Classifier()
# loss_FC_func = nn.CrossEntropyLoss()
# optimizer_FC = torch.optim.Adam(FC.parameters(), lr=0.001)


# ------------------------ 6、训练网络模型 ------------------------
c_t_pos = torch.zeros(1, 50)
c_t_neg = torch.zeros(1, 50)
out = 0

actor_loss = []
critic_loss = []
fc_loss = []
rights = []
for epoch in range(5):

    for i, data in enumerate(train_dataloader):
        # x, y --> [50, 1], [1,]
        x, y = data
        y = y.squeeze()
        # x: [32, 50] --> [50, 32]
        x = x.transpose(0, 1)

        h_a, c_a = critic.init_H_C()
        # h_c, c_c = critic.init_H_C()

        true_actions = []  # 真实动作
        pred_actions = []  # 预测动作
        actions_pair = []  # 回合预测动作对

        for j in range(x.shape[0]):
            # h_a = h_c
            # 选择动作
            # h_a, c_a, out = actor(x[j], h_a, c_a)
            h_a, c_a, out = critic(x[j], h_a, c_a)
            # out = torch.log_softmax(out, 1)
            # actions_pair.append(out)
            # pred_actions.append(torch.argmax(out, dim=1))
            #
            # if idx2word(x[j], vocab) in pos_dict_std:
            #     true_actions.append(0)
            # else:
            #     true_actions.append(1)

            # ------------- Critic更新参数 -------------------
            # 根据所推测的情感来判断使用哪个C通道, 0为正向标签
            # if torch.argmax(out, dim=1) == 0:
            #     h_c, c_t_pos, _ = critic(x[j], h_c, c_t_pos)
            # else:
            #     h_c, c_t_neg, _ = critic(x[j], h_c, c_t_neg)

        # 回合结束, 开始跟新参数
        # ------------------- 跟新Actor -------------------
        # true_actions = torch.LongTensor(true_actions)
        # # [50, 1]
        # pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
        # # 将含有tensor的list转化为tensor, [50, 3]
        # actions_pair = torch.cat(tuple(actions_pair), 0)
        #
        # H = F.cross_entropy(actions_pair, true_actions).item()
        # L_L_ = (true_actions == pred_actions.squeeze()).sum().item() / x.shape[0]
        #
        # R = H + 0.5 * L_L_
        # selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
        # loss_actor = selected_logprobs.mean()
        #
        # opti_actor.zero_grad()
        # loss_actor.backward( )
        # opti_actor.step()

        # ------------------- 跟新Critic -------------------
        # lo_critic = loss_critic(actions_pair, true_actions)
        #
        # opti_critic.zero_grad()
        # lo_critic.backward()
        # opti_critic.step()

        # ------------------- 跟新FC -------------------
        # end_output = FC(h_a)
        # end_output = torch.softmax(out, dim=1)
        loss_fc = loss_critic(out, y)

        opti_critic.zero_grad()
        loss_fc.backward()
        opti_critic.step()

        acc = get_accurate(out, y)
        rights.append(acc)
        fc_loss.append(loss_fc)

        if i % 50 == 0:
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('epoch: {}, 数据准确率: {:.5f}  loss: {:.5f}'.format(epoch + 1,right_ratio, torch.mean(torch.stack(fc_loss))))
            fc_loss = []
            rights = []
            # torch.save(model.state_dict(), save_path)
            # print("Save parameters")

# ------------------------ 7、测试网络模型 ------------------------









