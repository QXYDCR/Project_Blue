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


# ------------------- 从pos,neg文本(原始句子, 未被切分)中读取,并返回字典 -------------------
def read_data_from_pos_neg(pos_file, neg_file, is_filter = True):
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
    diction = {'PAD': 0}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
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
            return(w)
    return 'PAD'

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
    for word in vocab.keys():
        try:
            word_vecs[vocab[word]] = model[word]
            true_count_vec += 1
        except KeyError:
            pass

    print("词向量装载率: {0} / {1} == {2}".format(true_count_vec, len(vocab), 1.0 * true_count_vec / len(vocab)))
    return word_vecs


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
def convert_emotion_text_to_num(emotion_text, vocab, max_len = 50, is_filter = True):
    """
    :param emotion_text:
    :param vocab:
    :param max_len:
    :param is_filter:
    :return:
    """
    """
    :param data_path: 数据路径, txt
    :param vocab: 词表, w:i
    :param max_len: 每个句子映射成固定大小的长度，便于批训练
    :return: contents: 文本数值内容, label: 句子情感(0, 1)
    """
    contents, labels = [], []

    with open(emotion_text, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = jieba.lcut(line.strip())
            content = [vocab.get(w, 0) for w in words]
            # 截取最大长度
            content = content[:max_len]
            if len(content) < max_len:
                content += [vocab['PAD']] * (max_len - len(content))
            labels.append(1)
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

    return rights, len(label), 1.0 * rights / len(label)

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










