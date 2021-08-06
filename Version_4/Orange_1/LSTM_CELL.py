from Utils import *

"""
        Orange Version 2
Note:

"""

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'

BATCH = 1

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
train_data = dataset[5 * test_size :]
train_label = labels[5 * test_size :]
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)
train_dataset = TensorDataset(torch.from_numpy(train_data), torch.LongTensor(train_label))
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True, drop_last = True)


class Critic(nn.Module):
    def __init__(self, vocab_size=len(vocab), embedding_dim=50, hidden_size=50,
                 pretrained_embed=word2vec, is_updata_w2c=True, n_class=2, batch = BATCH):
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
loss_func_critic = nn.CrossEntropyLoss()
opti_critic = torch.optim.Adam(critic.parameters(), lr = 0.001)

# ------------------------ 6、训练网络模型 ------------------------

out = 0

actor_loss = []
critic_loss = []
fc_loss = []
rights = []


for epoch in range(8):
    for i, data in enumerate(zip(train_data, train_label)):
        # x, y --> [50, b], [b,]
        x, y = data
        x = torch.LongTensor(x).view(-1, 1)
        y = torch.LongTensor(y)

        h_c, c_c = critic.init_H_C()

        for j in range(x.shape[0]):
            h_c, c_c, out = critic(x[j], h_c, c_c)

        lo_critic = loss_func_critic(out, y)

        critic_loss.append(lo_critic)
        acc = get_accurate(out, y)
        rights.append(acc)

        opti_critic.zero_grad()
        lo_critic.backward()
        opti_critic.step()


        if i % 500 == 0:
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('epoch: {}, 数据准确率: {:.5f}  loss: {:.5f}'.format(epoch + 1,right_ratio,
                                    torch.mean(torch.stack(critic_loss))))
            fc_loss = []
            rights = []
            actor_loss = []
            critic_loss = []







