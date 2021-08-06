import torch
import torch.nn as nn
import torch.optim

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


def seg_word(sentence, fname, stop_file):
    """使用jieba对文档分词"""
    seg_list = jieba.cut(sentence)
    seg_result = []
    for w in seg_list:
        seg_result.append(w)
    # 读取停用词文件
    stopwords = set()
    with open(fname, 'r', encoding= 'utf-8') as fr:
        for word in fr:
            stopwords.add(word.strip())
    # 去除停用词
    return list(filter(lambda x: x not in stopwords, seg_result))


















