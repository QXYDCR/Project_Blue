import numpy as np
import re
import matplotlib.pyplot as plt
import jieba
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
# 加载机器学习的软件包
from sklearn.decomposition import PCA

#加载Word2Vec的软件包
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence



# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'
stop_words = ['的', '是', '就', '还', '就是', '在', '这个', '我', '都', '给', '和']
# stop_words = []

# ------------------------ 2、预处理数据 ------------------------

# 将文本中的标点符号过滤掉
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*()+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)


# 去掉多余的重复词语
def consics_word_by_stop(sentences, stop = stop_words):
    newline = []
    for word in sentences:
        if word in stop:
            continue
        else:
            newline.append(word)
    return newline


# 预处理数据
def process_data(good_file, bad_file, is_filter = True):
    all_words = []      # 存储所有的单词
    pos_sentences = []  # 存储正向的评论
    neg_sentences = []  # 存储负向的评论

    with open(good_file, 'r', encoding = 'utf-8') as f:
        for idx, line in enumerate(f):
            if is_filter:
                line = filter_punc(line)
            word_list = jieba.lcut(line)
            # 取出不重要的词语
            word_list = consics_word_by_stop(word_list, stop_words)
            if len(word_list) > 0:
                # all_words 是直接加载单词的后面，相当于不断地向右延伸
                # pos_sentences 是在下面加载句子的单词，相当于不断地向下延伸
                all_words += word_list
                pos_sentences.append(word_list)
    # data/good.txt 包含 8089 行, 100828 个词., dict, 3975
    print('{0} 包含 {1} 行, {2} 个词., dict, {3}'.format(good_file, idx + 1,
                            len(all_words), len(list(set(all_words)))))

    # 用于计算负向词典中长度
    pos_cunt = len(all_words)
    with open(bad_file, 'r', encoding = 'utf-8') as f:
        for idx, line in enumerate(f):
            if is_filter:
                line = filter_punc(line)
            word_list = jieba.lcut(line)
            word_list = consics_word_by_stop(word_list)
            if len(word_list) > 0:
                all_words += word_list
                neg_sentences.append(word_list)
    # data/good.txt 包含 5076 行, 56061 个词., new dict, 7134
    print('{0} 包含 {1} 行, {2} 个词., new dict, {3}'.format(bad_file, idx + 1,
                            len(all_words) - pos_cunt, len(list(set(all_words)))))

    # 建立词典
    dict = {}
    cnt = Counter(all_words)
    for word, frequence in cnt.items():
        dict[word] = [len(dict), frequence]
    dict = dict
    return dict, pos_sentences, neg_sentences

# 7126 8049 4982
dict, pos_sentences, neg_sentencs = process_data(good_file, bad_file)

# 得到正向词典、与负向词典
pos_dict = {}
temp_pos_word_list = []
for line in pos_sentences:
    temp_pos_word_list.extend(line)
cnt = Counter(temp_pos_word_list)
for word, frequence in cnt.items():
    pos_dict[word] = [len(dict), frequence]

# Positive number:  3967
# print("Positive number: ", len(pos_dict))

# 排序字典, 按照频率降序, 取前一千个元素
temp = []
temp = sorted(pos_dict.items(), key = lambda x: x[1][1], reverse= True)
temp = temp[:2000]
# 将列表转化为字典
pos_dict = {}
for i, w in temp:
    pos_dict[i] = w

# 新的字典 pos_dict含有1000个高频率的词

# 得到正向词典、与负向词典
neg_dict = {}
temp_neg_word_list = []
for line in neg_sentencs:
    temp_neg_word_list.extend(line)
cnt = Counter(temp_neg_word_list)
for word, frequence in cnt.items():
    neg_dict[word] = [len(dict), frequence]
print("Negative number: ", len(neg_dict))

# Negative number:  5034
# 排序字典, 按照频率降序, 取前一千个元素
temp = []
temp = sorted(neg_dict.items(), key = lambda x: x[1][1], reverse= True)
temp = temp[:2500]
# 将列表转化为字典
neg_dict = {}
for i, w in temp:
    neg_dict[i] = w



#根据单词返还单词的编码
def word2index(word, diction = dict):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value

#根据编码获得单词
def index2word(index, diction = dict):
    for w,v in diction.items():
        if v[0] == index:
            return(w)
    return(None)




# 调用Word2Vec的算法进行训练。
# 参数分别为：size: 嵌入后的词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(neg_sentencs + pos_sentences, vector_size = 40, window = 4 , min_count = 0, epochs= 5)
wv = model.wv
# print(len(wv))
# print(wv.get_vector('智子'))
# print(wv.most_similar('智子', topn=10))

# 遍历所有句子，将每一个词映射成编码
dataset = []            #数据集, 存放数字, (n, len(dict))
labels = []             #标签
sentences = []          #原始句子，调试用


# 把一句话中的number序列，转化为Word2Vec的数值
def series_to_word2vec(series):
    wordvec = []
    for num in series:
        if num == -1:
            continue
        vec = wv.get_vector(index2word(num))
        if vec in wv.vectors:
            wordvec.append(vec)
        # print(vec)
    return wordvec



# 处理正向评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in dict and len(new_sentence) < 50:
            new_sentence.append(word2index(l, dict))
    while len(new_sentence) < 50:
        new_sentence.append(-1)

    # print("sentence: ", len(new_sentence))
    dataset.append(new_sentence)
    labels.append(0)  # 正标签为0
    sentences.append(sentence)

# 处理负向评论
for sentence in neg_sentencs:
    new_sentence = []
    for l in sentence:
        if l in dict and len(new_sentence) < 50:
            new_sentence.append(word2index(l, dict))
    while len(new_sentence) < 50:
        new_sentence.append(-1)

    # print("sentence: ", len(new_sentence))
    dataset.append(new_sentence)
    labels.append(1)  # 负标签 1
    sentences.append(sentence)


# 将数字转化为单词
def idx_to_sentences(sentence):
    text = []
    for i in sentence:
        if i == -1:
            continue
        text.append(index2word(i))

    return text

# ['还', '蛮', '好', '的', '吧'] 1
# print(idx_to_sentences(dataset[10003]), labels[10003])



# ------------------------ 3、创建数据集 ------------------------

# 打乱所有的数据顺序，形成数据集
# indices为所有数据下标的一个全排列
indices = np.random.permutation(len(dataset))

# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

# 对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
# // 表示 int(a / b)
test_size = len(dataset) // 10
train_data = dataset[2 * test_size :]
train_label = labels[2 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 2 * test_size]
test_label = labels[test_size : 2 * test_size]

# ['不错', '穿着', '舒服', '还', '不贵', '挺', '好']
# idx_to_sentences(dataset[1])

# ----------------------------- 训练超参数 -----------------------------
LR_A = 0.001
ALPHA = 0.5
BELTA = 0.1
DROPOUT = 0.5

# ----------------------------- 4、模型定义 -----------------------------
class Actor_LSTM_Cell(nn.Module):
    def __init__(self):
        super(Actor_LSTM_Cell, self).__init__()
        out_class = 2
        self.LSTM_Actor = nn.LSTMCell(40, 60)
        self.fc = nn.Linear(60, out_class)

    def forward(self, x, h, c):
        h, c = self.LSTM_Actor(x, (h, c))
        c_n_class = self.fc(c)

        return h, c, c_n_class

# Define optimizer  Actor
Actor = Actor_LSTM_Cell()
optimizer_Actor = torch.optim.Adam(Actor.parameters(), lr = LR_A)



class FC_Classifier(nn.Module):
    def __init__(self):
        super(FC_Classifier, self).__init__()
        self.fc = nn.Linear(60, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


# Define Fc optimizer
FC = FC_Classifier()
optimizer_FC = torch.optim.Adam(FC.parameters(), lr=LR_A)
loss_FC = nn.CrossEntropyLoss()


class Critic_LSTM_Cell(nn.Module):
    def __init__(self):
        super(Critic_LSTM_Cell, self).__init__()
        out_class = 2
        self.LSTM_Critic = nn.LSTMCell(40, 60)
        self.fc = nn.Linear(60, out_class)

    def forward(self, x, h, c):
        h, c = self.LSTM_Critic(x, (h, c))
        c_n_class = self.fc(c)

        return h, c, c_n_class

# Define optimizer  Actor
Critic = Critic_LSTM_Cell()
optimizer_Critic = torch.optim.Adam(Critic.parameters(), lr = LR_A)
loss_Critic = nn.CrossEntropyLoss()


# ----------------------------- 5、训练过程 -----------------------------
for train_time in range(5):
    h0_actor = torch.randn(1, 60)
    c0_actor = torch.randn(1, 60)

    h0_critic = torch.randn(1, 60)
    c0_critic = torch.randn(1, 60)

    c0_pos = torch.randn(1, 60)
    c0_neg = torch.randn(1, 60)

    mat_loss_a = []

    for data, label in zip(train_data, train_label):
        sentence_idx = data
        a = idx_to_sentences(data)
        # print(idx_to_sentences(data), label, '<---------')
        sentence_vec = series_to_word2vec(data)
        sentence_vec = torch.FloatTensor(sentence_vec)
        # 初始化h0, h1, c0, c1, o, LSTMCell

        # 句子长度
        sen_len = sentence_vec.shape[0]

        # 定义状态
        s = h0_critic
        i = 0
        L_ = 0      # 正确预测词性个数
        L = len(a)  # 句子长度
        R = 0       # 回合奖励
        # 保存每回合所用的动作，动作概率，用于计算Actor的loss
        episode_action = []
        episode_pi     = []         # 每个单词的预测情感分布
        episode_true_action = []    # 每个单词的真实情感分布

        S_n = 0 # 记录最后一个状态
        for o in sentence_vec:

            # 词向量[1, 40]
            o = o.view(-1, 40)
            w = index2word(sentence_idx[i], dict)
            S_n = o

            # Actor 选择动作
            h0_actor = h0_critic

            h0_actor, c0_actor, n_class = Actor(o, h0_actor, c0_actor)
            n_class = torch.log_softmax(n_class, 1)
            episode_pi.append(n_class.detach().numpy())
            action = torch.max(n_class, dim = 1)[1].numpy()
            episode_action.append(action)

            # 记录预测的情感和实际情感是否相同
            if w in pos_dict and action == [0]:
                L_ += 1
            elif w in neg_dict and action == [1]:
                L_ += 1

            # 记录真实回合词语感情，用于计算Actor损失
            if w in neg_dict:
                episode_true_action.append(1)
            else:
                episode_true_action.append(0)

            # ------------- Critic更新参数 -------------------
            # 根据所推测的情感来判断使用哪个C通道, 0为正向标签
            if action == 0:
                h0_critic, c0_pos, _ = Critic(o, h0_critic, c0_pos)
            else:
                h0_critic, c0_neg, _  = Critic(o, h0_critic, c0_neg)
            i += 1


        # ------------- Actor更新参数 -------------------
        # [L, 1, 2] --> [L, 2]
        episode_pi = torch.FloatTensor(episode_pi)
        episode_pi = episode_pi.view(-1, 2)

        episode_action = torch.LongTensor(episode_action)

        episode_true_action = torch.LongTensor(episode_true_action)

        H = F.cross_entropy(episode_pi, episode_true_action)
        R = -ALPHA * 1.0 * (L_ / L) + H

        # log(π(a|s) + ... log(
        selected_logprobs = R * torch.gather(episode_pi, 1,  episode_action).squeeze()

        loss_actor = selected_logprobs.mean()
        loss_actor = torch.tensor(loss_actor, requires_grad = True)
        optimizer_Actor.zero_grad()
        loss_actor.backward()
        optimizer_Actor.step()
        # print("Loss_A: ", loss_actor)

        # ------------- Critic更新参数 -------------------

        h0_critic, c0_critic, pre_fc_label = Critic(S_n, h0_critic, c0_critic)
        pre_fc_label = torch.softmax(pre_fc_label, dim=1)
        label = torch.LongTensor([label])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        loss_c = loss_Critic(pre_fc_label, label)
        # print("loss_Critic: ", loss_c)

        loss_c = torch.tensor(loss_c, requires_grad = True)
        optimizer_Critic.zero_grad()
        loss_c.backward()
        optimizer_Critic.step()

        # ------------- 全连接层更新参数 -------------------
        pre_fc_label = FC(h0_critic)
        pre_fc_label = torch.softmax(pre_fc_label, dim = 1)

        label = torch.LongTensor([label])
        loss_fc = loss_FC(pre_fc_label, label)
        loss_fc = torch.tensor(loss_fc, requires_grad=True)
        # print("loss_fc: ", loss_fc)

        optimizer_FC.zero_grad()
        loss_fc.backward()
        optimizer_FC.step()

        mat_loss_a.append(loss_fc.detach().numpy())


    plt.plot(range(len(mat_loss_a)), mat_loss_a)
    plt.show()

# 模型保存


# 模型加载


# 进行测试





