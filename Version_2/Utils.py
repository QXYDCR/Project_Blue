#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
#from torch.autograd import Variable

# 自然语言处理相关的包
import re #正则表达式的包
import jieba #结巴分词包
from collections import Counter #搜集器，可以让统计词频更简单

import numpy as np
import torch
import torch.nn.functional as F

#绘图、计算用的程序包
import matplotlib.pyplot as plt



"""
    定义工具类，用于简化代码、使代码具有可读性
"""


# 将文本中的标点符号过滤掉
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", sentence)
    return(sentence)


# 扫描所有的文本，分词、建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(good_file, bad_file, is_filter = True):
    all_words = []      #存储所有的单词
    pos_sentences = []  #存储正向的评论
    neg_sentences = []  #存储负向的评论
    with open(good_file, 'r', encoding= 'utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(good_file, idx+1, len(all_words)))

    count = len(all_words)
    with open(bad_file, 'r', encoding= 'utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(bad_file, idx+1, len(all_words)-count))

    #建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}

    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小：{}'.format(len(diction)))

    return pos_sentences, neg_sentences, diction

#根据单词返还单词的编码
def word2idx(word, vocab):
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return(value)

#根据编码获得单词
def idx2word(index, vocab):
    for w,v in vocab.items():
        if v[0] == index:
            return(w)
    return(None)


# 将句子变为词袋模型，词语之间没有顺序, 并且归一化数据
def sentence2vec(sentence, vocab):
    vector = np.zeros(len(vocab))
    for l in sentence:
        vector[l] += 1
    return(1.0 * vector / len(sentence))


# 将文本转化为向量，以句子为单位
def text_to_vec(pos, neg, vocab):
    dataset = []
    labels = []
    for lines in pos:
        sentence_idx = []
        for w in lines:
            if w in vocab:
                sentence_idx.append(word2idx(w, vocab))
        dataset.append(sentence2vec(sentence_idx, vocab))
        labels.append(0)

    for lines in neg:
        sentence_idx = []
        for w in lines:
            if w in vocab:
                sentence_idx.append(word2idx(w, vocab))
        dataset.append(sentence2vec(sentence_idx, vocab))
        labels.append(1)
    return dataset, labels

def text_to_vec_word2vec(pos_sentences, neg_sentences, vocab):
    dataset = []
    labels = []
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

    return dataset, labels

# 将句子的索引转化为单词
def idx_to_sentence(idx, vocab):
    sentences = []
    for i in idx:
        sentences.append(idx2word(i, vocab))
    return sentences



# 用于调试，作为标记语句
def mark_star(mark = '*', num = 20):
    print(mark * num)

# 将数据集按照比率划分 三个部分: train_data, valid_data, test_data
def split_dataset(datset, split_ration = [6, 1, 3]):
    unit_size  = len(datset) // 10
    train_data = datset[: split_ration[0] * unit_size]
    valid_data = datset[split_ration[0] * unit_size : (split_ration[0] + split_ration[1]) * unit_size]
    test_data  = datset[-split_ration[2] * unit_size :]

    return train_data, valid_data, test_data


# 给出两个列表或者tensor, 得出正确率
# y: [b, N], label: [b]
def get_accurate(y, label, is_tensor = True):
    if is_tensor == False:
        y = torch.FloatTensor(y)
        label = torch.FloatTensor(label)

    pred = torch.max(y, dim = 1)[1]
    rights = pred.eq(label.data.view_as(pred)).sum()

    return rights, len(label)



# 根据数据集合标签，建立情感词典
def get_emotion_dict(dataset, label, vocab):
    pos_dict = {}
    neg_dict = {}

    for lines in zip(dataset, label):
        sen, flag = lines
        for idx in sen:
            if flag == True:
                pos_dict[idx2word(idx, vocab)] = len(pos_dict) + len(neg_dict)
            else:
                neg_dict[idx2word(idx, vocab)] = len(pos_dict) + len(neg_dict)

        # print('pos: {0} neg: {1} 行.'.format(len(pos_dict), len(neg_dict)))
    return pos_dict, neg_dict















