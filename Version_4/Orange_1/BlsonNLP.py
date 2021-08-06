import torch

from Utils import *

"""
        Orange Version 2
Note:

"""
from Demo import Sentiment_Score

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

score_file = 'C:/Users/Chihiro/Desktop/RL Paper\Project\Project_Blue_1/' \
             '数据集/Chinese_Corpus-master/sentiment_dict/sentiment_dict/sentiment_score.txt'
def load_dict_scores(file = score_file):
    """
    :param file: (str), 词语分数文件
    :return: (dict), 词语分数字典
    """
    dict = {}
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        for line in lines:
            line = line.strip().split(' ')
            try:
                dict[line[0]] = np.float(line[1])
            except:
                pass
    return dict

word_socre = load_dict_scores(score_file)



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

mark_star()
s = Sentiment_Score()
for i, data in enumerate(zip(train_data, train_label)):
    # x, y --> [30, 1], [1,]
    x, y = data
    ff = convert_num_to_txt(x, vocab)



mark_star()

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]


# ------------------------ 5、定义网络模型 ------------------------

class Actor(nn.Module):
    def __init__(self, vocab_size=len(vocab), embedding_dim=50, hidden_size=50,
                 pretrained_embed=word2vec, is_updata_w2c=False, n_class=3):
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

        # 第一优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.NLLLoss(reduction = 'none')

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

class FC_Classifier(nn.Module):
    def __init__(self, hidden_size = 50, n_class = 2):
        super(FC_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.fc = nn.Linear(hidden_size, n_class)

        self.loss_FC_func = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.fc(x)
        return x


# ------------------------ 6、训练网络模型 ------------------------
actor = Actor()
fc = FC_Classifier()

Ac_demo = Actor()

actor_loss = []
critic_loss = []
fc_loss = []
rights = []
reward = []

EPSILON = 0.2
for epoch in range(8):
    EPSILON = 0.2 - (epoch) * 0.2 / 8

    for i, data in enumerate(zip(train_data, train_label)):
        # x, y --> [30, 1], [1,]
        x, y = data
        ff = convert_num_to_txt(x, vocab)
        x = torch.LongTensor(x).view(-1, 1)
        y = torch.LongTensor(y)

        true_actions = []  # 真实动作
        pred_actions = []  # 预测动作
        actions_pair = []  # 回合预测动作对

        h, c = actor.init_H_C()
        out = 0
        I = []             # 词语级情感评分

        for j in range(x.shape[0]):
            h, c, out = actor(x[j], h, c)

            out = torch.log_softmax(out, 1)
            actions_pair.append(out)
            pred_actions.append(torch.argmax(out, dim=1))

            w = idx2word(x[j], vocab)
            # e-greedy
            if np.random.rand() > EPSILON:
                if w in pos_dict_std:
                    true_actions.append(0)
                elif w in neg_dict_std:
                    true_actions.append(1)
                else:
                    true_actions.append(2)
            else:
                true_actions.append(np.random.randint(0, 3))

            # 得到当前词语的情感评分
            # w = idx2word(x[j], vocab)
            s = word_socre.get(idx2word(x[j], vocab), 0)
            if s > 0:
                # true_actions.append(0)
                s = s * 0.35
            # elif s < 0:
            #     true_actions.append(1)
            # else:
            #     true_actions.append(1)
            I.append(s)

        # 回合更新参数
        actor.optimizer.zero_grad()
        fc.optimizer.zero_grad()

        true_actions = torch.LongTensor(true_actions)
        # [30, 1]
        pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
        # 将含有tensor的list转化为tensor, [32, 2]
        actions_pair = torch.cat(tuple(actions_pair), 0)

        # 计算回合的得分, 每个步骤都有分
        scores = get_episode_score2(I, pred_actions.numpy())
        # scores = discount_and_norm_rewards(scores)
        reward.append(scores.sum())
        # selected_logprobs = torch.gather(actions_pair, 1, pred_actions).squeeze()
        selected_logprobs = actor.criterion(actions_pair, pred_actions.squeeze())

        loss_actor = -(scores *  selected_logprobs).mean()
        # print(scores)
        # 更新FC
        fc_func = fc.loss_FC_func(out, y)

        # 保存值
        acc = get_accurate(out, y)
        rights.append(acc)

        actor_loss.append(loss_actor)
        fc_loss.append(fc_func)

        loss_actor.backward(retain_graph = True)
        fc_func.backward()
        # bp
        actor.optimizer.step()
        fc.optimizer.step()

        if i % 400 == 0:
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('epoch: {},\t数据准确率: {:.5f}\tfc_loss: {:.5f}\tactor_loss: {:.5f}\treward: {:.5f}'.
                                    format(epoch + 1,right_ratio,
                                    torch.mean(torch.stack(fc_loss)),
                                    torch.mean(torch.stack(actor_loss)),
                                           np.mean(reward)))

            fc_loss = []
            rights = []
            actor_loss = []
            critic_loss = []
            reward = []

# ------------------------ 7、测试网络模型 ------------------------


actor_2 = Actor()

def train_adaboost():
    actor_loss = []
    fc_loss = []
    rights = []
    reward = []

    for epoch in range(4):
        for i, data in enumerate(zip(train_data, train_label)):
            # x, y --> [30, 1], [1,]
            x, y = data
            ff = convert_num_to_txt(x, vocab)
            x = torch.LongTensor(x).view(-1, 1)
            y = torch.LongTensor(y)

            true_actions = []  # 真实动作
            pred_actions = []  # 预测动作
            actions_pair = []  # 回合预测动作对

            label_accu = 0

            h, c = actor.init_H_C()
            h2, c2 = actor_2.init_H_C()
            out2 = 0
            out = 0
            I = []  # 词语级情感评分

            for j in range(x.shape[0]):
                h, c, out = actor(x[j], h, c)
                h2, c2, out2 = actor_2(x[j], h2, c2)

                # 得到当前词语的情感评分
                s = word_socre.get(idx2word(x[j], vocab), 0)
                if s > 0:
                    true_actions.append(0)
                    s = s * 0.4
                elif s < 0:
                    true_actions.append(1)
                else:
                    true_actions.append(2)

                I.append(s)

            # 回合更新参数
            actor.optimizer.zero_grad()
            fc.optimizer.zero_grad()

            if np.sum(I) > 0:
                label_accu = 0
            elif np.sum(I) < 0:
                label_accu = 1
            else:
                label_accu = 2

            # 更新FC, 使用另外一个out2
            fc_func = fc.loss_FC_func(out2, y)

            # 保存值
            acc = get_accurate(out2, y)
            rights.append(acc)

            fc_loss.append(fc_func)

            fc_func.backward()

            # bp
            fc.optimizer.step()

            if i % 400 == 0:
                right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                print('epoch: {},\t数据准确率: {:.5f}\tfc_loss: {:.5f}}'.
                      format(epoch + 1, right_ratio,
                             torch.mean(torch.stack(fc_loss))))
                fc_loss = []
                rights = []

# train_adaboost()

def train_2_self_score():
    actor_loss = []
    fc_loss = []
    rights = []
    reward = []

    for epoch in range(8):
        EPSILON = 0.2 - (epoch) * 0.2 / 8
        for i, data in enumerate(zip(train_data, train_label)):
            # x, y --> [30, 1], [1,]
            x, y = data
            ff = convert_num_to_txt(x, vocab)
            x = torch.LongTensor(x).view(-1, 1)
            y = torch.LongTensor(y)

            true_actions = []  # 真实动作
            pred_actions = []  # 预测动作
            actions_pair = []  # 回合预测动作对

            h, c = actor.init_H_C()
            out = 0
            I = []             # 词语级情感评分

            for j in range(x.shape[0]):
                h, c, out = actor(x[j], h, c)

                out = torch.log_softmax(out, 1)
                actions_pair.append(out)
                pred_actions.append(torch.argmax(out, dim=1))

                w = idx2word(x[j], vocab)
                # e-greedy
                if np.random.rand() > EPSILON:
                    if w in pos_dict_std:
                        true_actions.append(0)
                    elif w in neg_dict_std:
                        true_actions.append(1)
                    else:
                        true_actions.append(2)
                else:
                    true_actions.append(np.random.randint(0, 3))

                # 得到当前词语的情感评分
                # w = idx2word(x[j], vocab)
                # s = word_socre.get(idx2word(x[j], vocab), 0)
                # if s > 0:
                #     # true_actions.append(0)
                #     s = s * 0.35
                # elif s < 0:
                #     true_actions.append(1)
                # else:
                #     true_actions.append(1)
                # I.append(s)
            ff = s.regular_text(ff)
            I, _ = s.get_sentence_scores(ff, vocab)
            for jj in I:
                if jj > 0:
                    jj *= 0.4

            # 回合更新参数
            actor.optimizer.zero_grad()
            fc.optimizer.zero_grad()

            true_actions = torch.LongTensor(true_actions)
            # [30, 1]
            pred_actions = torch.LongTensor(pred_actions).unsqueeze(1)
            # 将含有tensor的list转化为tensor, [32, 2]
            actions_pair = torch.cat(tuple(actions_pair), 0)

            # 计算回合的得分, 每个步骤都有分
            scores = get_episode_score2(I, pred_actions.numpy())
            # scores = discount_and_norm_rewards(scores)
            reward.append(scores.sum())
            # selected_logprobs = torch.gather(actions_pair, 1, pred_actions).squeeze()
            selected_logprobs = actor.criterion(actions_pair, pred_actions.squeeze())

            loss_actor = (scores *  selected_logprobs).mean()
            # print(scores)
            # 更新FC
            fc_func = fc.loss_FC_func(out, y)

            # 保存值
            acc = get_accurate(out, y)
            rights.append(acc)

            actor_loss.append(loss_actor)
            fc_loss.append(fc_func)

            loss_actor.backward(retain_graph = True)
            fc_func.backward()
            # bp
            actor.optimizer.step()
            fc.optimizer.step()

            if i % 400 == 0:
                right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                print('epoch: {},\t数据准确率: {:.5f}\tfc_loss: {:.5f}\tactor_loss: {:.5f}\treward: {:.5f}'.
                                        format(epoch + 1,right_ratio,
                                        torch.mean(torch.stack(fc_loss)),
                                        torch.mean(torch.stack(actor_loss)),
                                               np.mean(reward)))

                fc_loss = []
                rights = []
                actor_loss = []
                critic_loss = []
                reward = []

# train_2_self_score()

