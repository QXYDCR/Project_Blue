#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
#from torch.autograd import Variable

# 自然语言处理相关的包
import re #正则表达式的包
import jieba #结巴分词包
from collections import Counter #搜集器，可以让统计词频更简单
import gensim
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
#绘图、计算用的程序包
import matplotlib.pyplot as plt
"""
    工具类的定义
"""

# ------------------- 将文本中的标点符号过滤掉 -------------------
def filter_punc(sentence):
    """
    :param sentence: 将指定的句子进行过滤标点
    :return: 返回过滤非字母数字后的句子
    """
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)


file_path = 'C:/Users/Chihiro/Desktop/RL Paper/Project/Project_Blue_1' \
            '/数据集/情感极性词典/snownlp_stopwords.txt'
def load_stopwords(fname = file_path):
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for word in f:
            stopwords.add(word.strip())

    return stopwords

# ------------------- 从pos,neg文本(原始句子, 未被切分)中读取,并返回字典 -------------------
def read_data_from_pos_neg(pos_file, neg_file, is_filter = True, is_freq = False):
    """
    :param pos_file: positive 文件名
    :param neg_file: negative 文件名
    :param is_filter:   是否过滤标点符号
    :return: 正向语句集合、负向语句集合、词表
    """
    all_words = []      #存储所有的单词
    pos_sentences = []  #存储正向的评论
    neg_sentences = []  #存储负向的评论
    with open(pos_file, 'r', encoding= 'utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0}文件包含 {1} 行, {2} 个词.'.format(pos_file, idx + 1, len(all_words)))

    count = len(all_words)
    with open(neg_file, 'r', encoding= 'utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0}文件包含 {1} 行, {2} 个词.'.format(neg_file, idx + 1, len(all_words)-count))

    #建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {'_PAD_': 0}
    cnt = Counter(all_words)
    if is_freq == True:
        for word, freq in cnt.items():
            diction[word] = [len(diction), freq]
    else:
        for word, freq in cnt.items():
            diction[word] = len(diction)
    print('字典大小：{}'.format(len(diction)))

    return pos_sentences, neg_sentences, diction


# ------------------- 从切分文本中读取,标签，内容在一起,并返回字典 -------------------
def read_build_vocab(paths = ['./Dataset/train.txt', './Dataset/validation.txt'], is_freq = False):
    """
    :param paths: 读取已经被切分好的数据集, 类型为list
    :param is_freq: 是否需要加入词频在字典中
    :return: 返回词表
    """
    diction = {'PAD': 0}
    # paths = ['./Dataset/train.txt', './Dataset/validation.txt']
    all_words = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    all_words.append(word)

    cnt = Counter(all_words)
    if is_freq == True:
        for word, freq in cnt.items():
            diction[word] = [len(diction), freq]
    else:
        for word, freq in cnt.items():
            diction[word] = len(diction)
    print('字典大小：{}'.format(len(diction)))

    return diction

# ------------------- 建立数字-->文本字典 -------------------
def idx2word(index, vocab):
    """
    :param index: 单词对应的索引
    :param vocab: 词表
    :return: 返回索引对应的单词
    """
    for w,v in vocab.items():
        if index == v:
            return w
    return '_PAD_'

# ------------------- 通过索引组来获取数据组 -------------------
def idxs_2_words(idxs, vocab, dict):
    """
    :param idxs:  索引组,[idx, ]
    :param vocab: 词表
    :param dict: 极性词典
    :return: 单词列表, 是否在极性词典中
    """

    # 将数字转化为词
    words = []
    for idx in idxs:
        flag = False
        a = idx2word(idx, vocab)
        for w, v in vocab.items():
            if idx == v:
                words.append(w)
                flag = True
        if flag == False:
            words.append('_PAD_')

    actions = []
    for w in words:
        if w in dict:
            actions.append(0)
        else:
            actions.append(1)

    return words, actions



# ------------------- 建立单词的词向量映射 -------------------
# 加载预定义的词向量, 并将每一个单词映射成预向量中存在的向量
def build_word2vec(pre_word2vec, vocab):
    """
    :param pre_word2vec: 预定义的词向量文件，后缀名bin
    :param vocab: 词汇映射表
    :return: i : vec, 返回索引: 预定义词向量的词表
    """
    n_vocab = len(vocab)
    model = gensim.models.KeyedVectors.load_word2vec_format(pre_word2vec, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_vocab, model.vector_size]))

    # 判断自检语料库中是否在预加载的词向量中，如果是则加入
    true_count_vec = 0
    i = 0
    for word in vocab.keys():
        try:
            word_vecs[vocab[word]] = model[word]
            true_count_vec += 1
        except KeyError:
            pass

    print("词向量装载率: {0} / {1} == {2}".format(true_count_vec, len(vocab), 1.0 * true_count_vec / len(vocab)))
    return word_vecs


# 加载现有的词语极性词典
def load_dict(file):
    """
    :param file: (str), 文件名
    :return: (dict), 极性词典
    """
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        dictionary = [word.strip() for word in lines]
    dict = {}
    for i, w in enumerate(dictionary):
        dict[w] = i
    return dict


# ------------------- 将文本转化为数字 -------------------
# 将文本转化为固定大小维度的数字组合(来源于词表), 多余截断，少的填充
def convert_txt_to_num(data_path, vocab, max_len = 50):
    """
    :param data_path: 数据路径, txt
    :param vocab: 词表, w:i
    :param max_len: 每个句子映射成固定大小的长度，便于批训练
    :return: contents: 文本数值内容, label: 句子情感(0, 1)
    """
    contents, labels = [], []

    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [vocab.get(w, 0) for w in sp[1:]]
            # 截取最大长度
            content = content[ :max_len]
            if len(content) < max_len:
                content += [vocab['PAD']] * (max_len - len(content))
            labels.append(label)
            contents.append(content)

        counter = Counter(labels)
        print('总样本数为：%d' % (len(labels)))
        print('各个类别样本数如下：')
        for w in counter:
            print(w, counter[w])

        contents = np.asarray(contents)
        labels = np.array([[int(l)] for l in labels])

        return contents, labels


# ------------------- 将文本转化为数字 -------------------
# 将已分割好的文本转化为固定大小维度的数字组合(来源于词表), 多余截断，少的填充
# 此为上面函数的重载, 文本中只有数据，文本为单一词性
def convert_emotion_text_to_num(emotion_text, vocab, max_len = 50, is_filter = True, lnum = 0):
    """
    :param emotion_text: 极性文本词典文件
    :param vocab: 自建词表
    :param max_len: 每句话表示的最大长度
    :param is_filter: 是否过滤标点符号
    :return: 数值形式的文本内容和标签
    """

    contents, labels = [], []
    stop_words = load_stopwords()

    with open(emotion_text, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = []
            pre_words = jieba.lcut(line.strip())

            # 过滤停顿词
            for w in pre_words:
                if w not in stop_words:
                    words.append(w)

            content = [vocab.get(w, 0) for w in words]
            # 截取最大长度
            content = content[:max_len]
            if len(content) < max_len:
                content += [vocab['_PAD_']] * (max_len - len(content))
            labels.append(lnum)
            contents.append(content)

        contents = np.asarray(contents)
        labels = np.array([[l] for l in labels])

        return contents, labels



# 将文本数值转化为文本
def convert_num_to_txt(sentnece, vocab):
    """
    :param sentnece: 数值化的句子
    :param vocab: 词表
    :return: 返回单词化的句子
    """
    content = []
    for idx in sentnece:
        content.append(idx2word(idx, vocab))
    return content

# ------------------- 得到准确率 -------------------
def get_accurate(y, label):
    """
    :param y: 预测值
    :param label: 真实标签值
    :return: 分类正确数、样本数、准确率
    """
    pred_y = torch.argmax(y, dim = 1)
    rights = pred_y.eq(label.data.view_as(pred_y)).sum().item()

    return rights, len(label)

# ------------------- 得到准确率 -------------------
def test_predict(dataloader, model):
    # 测试过程
    count, correct = 0, 0
    for _, (batch_x, batch_y) in enumerate(dataloader):
        h, c = model.init_H_C()
        output, h, c = model(batch_x, h, c)
        correct += (output.argmax(1) == batch_y.squeeze()).sum().item()
        count += len(batch_x)

    # 打印准确率
    print('test accuracy is {:.2f}%.'.format(100. * correct / count))


# -------------------标记语言, 用于分隔文本等 -------------------
def mark_star(left_mark = '=', right_mark = '=', num = 25):
    print(left_mark * num, right_mark * num)


def mix_dataset(pos_data, pos_lab, neg_data, neg_lab):
    # 混合训练集和验证集
    dataset = np.vstack([pos_data, neg_data])
    labels = np.concatenate([pos_lab, neg_lab])

    return dataset, labels

def random_data(dataset, labels):
    indices = np.random.permutation(len(dataset))
    dataset = [dataset[i] for i in indices]
    labels = [labels[i] for i in indices]

    return dataset, labels


def get_score(I, idx, E):
    for i in range(len(E)):
        if E[i] == 0:
            E[i] = 0
        elif E[i] == 1:
            E[i] = 1
        else:
            E[i] = -1

    size = 30
    C = np.zeros(size)
    ALPHA = 0.8
    S_V = 6

    for i in range(size):
        if i == 0:
            C[0] = I[0]
        else:
            if I[i] == 0:
                C[i] = ALPHA * C[i - 1]
            else:
                C[i] = C[i - 1] + I[i]

    score = 0
    # 给定当前位置，如何Ct-1 == 0
    if idx == 0:
        score = I[0]
    elif C[idx - 1] == 0:   # 1
        if E[idx] == 0:  # 判断当前的词的情感[-1, 0, 1]
            score = -np.abs(I[idx])
        else:
            score = I[idx] * E[idx]
    elif C[idx - 1] * I[idx] > 0:   # 2
        if E[idx] * C[idx - 1] > 0:
            score = S_V
        elif E[idx] * C[idx - 1] < 0:
            score = -S_V
        else:
            score = -np.abs(I[idx])
    else:
        if E[idx] * C[idx - 1] >= 0:
            score = S_V - np.abs(I[idx])
        else:
            score = (S_V - np.abs(I[idx])) * np.abs(I[idx])

    score = np.clip(score, -6, 6) / 6
    return score


def get_episode_score(I, A):
    size = 30
    std = 6
    # A = C
    # 更改A的动作搭配
    for i in range(30):
        if A[i] == 0:    # 积极情感
            A[i] = 1
        elif A[i] == 1:  # 消极情感:
            A[i] = -1
        else:
            A[i] = 0     # 中性

    scores = np.zeros(size)
    Q = np.zeros(size)


    ALPHA = 1.2
    gamma = 0.95
    # 计算句子累积情感Q值
    for i in range(30):
        if i == 0:      #如果i = 0时, 单独考虑
            Q[i] = I[i]
        else:
            if I[i] == 0:  # 如果当前词的BlsonNLP值为0，即代表中性
                Q[i] = ALPHA * Q[i - 1]
            else:  # 当前词拥有正负的含义
                Q[i] = I[i] + Q[i - 1] * gamma

    # 计算词语的得分:
    # A: [0, 1, -1], 积极、消极、中性
    for i in range(30):
        if i == 0:      #如果i = 0时, 单独考虑
            scores[i] = Q[i]
        else:
            if Q[i - 1] == 0:       # 如果前面句子为中性
                if A[i] == 0:       # 如果当前动作为中性
                    scores[i] = -np.abs(I[i])
                else:
                    scores[i] = I[i] * A[i]
            elif Q[i - 1] * I[i] > 0:   # 如果前面句子和当前词语的感情相同，即标签正确
                if Q[i - 1] * A[i] > 0: # 预测正确
                    scores[i] = std
                elif  Q[i - 1] * A[i] < 0:  # 预测错误
                    scores[i] = -std
                else:
                    scores[i] = -np.abs(I[i])
            elif Q[i - 1] * I[i] < 0:  # 如果前面句子和当前词语的感情相反，根据大小改表情感
                if Q[i - 1] * A[i] >= 0:  # 预测正确
                    scores[i] = std - np.abs(I[i])
                else:
                    scores[i] = (np.abs(I[i]) - np.abs(Q[i - 1]))* np.abs(I[i])



    # 归一化处理
    scores = torch.tensor(scores) / 6
    # scores -= torch.mean(scores)
    # scores /= torch.std(scores)

    # 更改A的动作搭配
    for i in range(30):
        if A[i] == 1:  # 积极情感
            A[i] = 0
        elif A[i] == -1:  # 消极情感:
            A[i] = 1
        else:
            A[i] = 2  # 中性

    return scores


def get_episode_score2(I, A):
    size = 30
    std = 6
    # A = C
    # 更改A的动作搭配
    for i in range(size):
        if A[i] == 0:    # 积极情感
            A[i] = 1
        elif A[i] == 1:  # 消极情感:
            A[i] = -1
        else:
            A[i] = 0     # 中性

    scores = np.zeros(size)
    Q = np.zeros(size)

    ALPHA = 1.2
    gamma = 0.9
    # 计算句子累积情感Q值
    for i in range(size):
        if i == 0:      #如果i = 0时, 单独考虑
            Q[i] = I[i]
        else:
            if I[i] == 0:  # 如果当前词的BlsonNLP值为0，即代表中性
                Q[i] = ALPHA * Q[i - 1]
            else:  # 当前词拥有正负的含义
                Q[i] = I[i] + Q[i - 1] * gamma

    # 计算词语的得分:
    # A: [0, 1, -1], 积极、消极、中性
    for i in range(size):
        if i == 0:      #如果i = 0时, 单独考虑
            scores[i] = Q[i]
        else:
            if Q[i - 1] == 0:       # 如果前面句子为中性
                if A[i] == 0:       # 如果当前动作为中性
                    # scores[i] = -np.abs(I[i])
                    scores[i] = np.abs(I[i])
                else:
                    scores[i] = I[i] * A[i]
            elif Q[i - 1] * I[i] > 0:   # 如果前面句子和当前词语的感情相同，即标签正确
                if Q[i - 1] * A[i] > 0: # 预测正确
                    scores[i] = std
                elif  Q[i - 1] * A[i] < 0:  # 预测错误
                    scores[i] = -std
                else:
                    scores[i] = -np.abs(I[i])
            elif Q[i - 1] * I[i] < 0:  # 如果前面句子和当前词语的感情相反，根据大小改表情感
                if Q[i - 1] * A[i] >= 0:  # 预测正确
                    scores[i] = std - np.abs(I[i])
                else:
                    scores[i] = (np.abs(I[i]) - np.abs(Q[i - 1]))* np.abs(I[i])
                # if Q[i] * A[i] > 0:
                #     scores[i] = np.abs(I[i])
                # else:
                #     scores[i] = (np.abs(I[i]) - np.abs(Q[i - 1])) * np.abs(I[i])


    # 归一化处理
    scores = torch.tensor(scores) / std
    # scores -= torch.mean(scores)
    # scores /= torch.std(scores)

    # 更改A的动作搭配
    for i in range(size):
        if A[i] == 1:  # 积极情感
            A[i] = 0
        elif A[i] == -1:  # 消极情感:
            A[i] = 1
        else:
            A[i] = 2  # 中性

    return scores



def discount_and_norm_rewards(scores):
    # discount episode rewards
    # generate a array with the same type and shape as given array
    discounted_ep_rs = np.zeros_like(scores)
    running_add = 0
    for t in range(0, len(scores)):
        running_add = running_add * 1 + scores[t]
        discounted_ep_rs[t] = running_add


    return torch.FloatTensor(discounted_ep_rs)




