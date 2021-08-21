import pandas as pd
import jieba
import re
from gensim.models import word2vec


# 分词
def tokenizer(text):
    """"""
    return [word for word in jieba.lcut(text) if word not in stop_words]

# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt',encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

traindata = pd.read_csv('data/train.tsv', sep='\t')
validata = pd.read_csv('data/validation.tsv', sep='\t')
print(traindata.head())

text = pd.read_csv('data/text.tsv')

stop_words = get_stop_words()

text_cut = []
i = 0
for row in text.itertuples():
    seg = tokenizer(row[1])
    text_cut.append(seg)
    i += 1
print(text_cut[0:5])
print("len: ", i)










