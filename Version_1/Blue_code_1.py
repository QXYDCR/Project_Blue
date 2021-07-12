import numpy as np
import re
import matplotlib.pyplot as plt
import jieba
import torch
import torch.nn as nn
import torch.optim
from collections import Counter
import torch.nn.functional as F
# 加载机器学习的软件包
from sklearn.decomposition import PCA

#加载Word2Vec的软件包
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence
"""

    Project Document
    
"""

# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'
# stop_words = ['的', '是', '就', '还', '就是', '在', '这个', '我']
stop_words = []

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

# 建立词到索引的转换
def word_to_idx(word, dict = dict):
    if word in dict:
        value = dict[word][0]
    else:
        value = -1
    return value

# 建立索引到词的转换
def idx_to_word(idx, dict = dict):
    for w, v in dict.items():
        if v[0] == idx:
            return w
    return None


# ------------------------ 3、词袋模型的建立 ------------------------

# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
def sentence_to_vec(sentence, dict):
    vec = np.zeros(len(dict))
    for i in sentence:
        vec[i] += 1
    # 归一化处理，提高速度
    return 1.0 * vec / len(sentence)

# 遍历所有句子，将每一个词映射成编码
dataset = []            #数据集, 存放数字, (n, len(dict))
labels = []             #标签
sentences = []          #原始句子，调试用



# 处理正向评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in dict:
            new_sentence.append(word_to_idx(l, dict))
    dataset.append(sentence_to_vec(new_sentence, dict))
    labels.append(0)  # 正标签为0
    sentences.append(sentence)

# 处理负向评论
for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in dict:
            new_sentence.append(word_to_idx(l, dict))
    dataset.append(sentence_to_vec(new_sentence, dict))
    labels.append(1)  # 正标签为0
    sentences.append(sentence)

# 打乱所有的数据顺序，形成数据集
# indices为所有数据下标的一个全排列
indices = np.random.permutation(len(dataset))
# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

# 对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
# // 6 : 1 : 3
test_size = len(dataset) // 10
train_data = dataset[4 * test_size :]
train_label = labels[4 * test_size :]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size : 4 * test_size]
test_label = labels[test_size : 4 * test_size]




# ----------------------------- 模型定义 -----------------------------
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()







