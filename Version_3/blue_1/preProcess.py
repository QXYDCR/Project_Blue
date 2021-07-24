from __future__ import unicode_literals, print_function, division
from io import open
import torch
import re
import  numpy as np
import gensim
from torch.utils.data import Dataset
from Config import Config

class MyDataset(Dataset):
    def __init__(self, Data, Label):
        super(MyDataset, self).__init__()
        self.Data = Data
        if Label is not None:
            self.Label = Label

    def __int__(self):
        return len(self.Data)

    def __getitem__(self, index):
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])
            label = torch.from_numpy(self.Label[index])
            return data, label
        else:
            data = torch.from_numpy(self.Data[index])
            return data

#创建停用词表
def stopwordslist():
    stopwords = [line.strip() for line in open('./word2vec_data/stopwords.txt'
                                               ,encoding='UTF-8').readlines()]
    return stopwords

def build_word2id(file):
    # file : 文件保存地址
    stopwords = stopwordslist()
    paths = [Config.train_path, Config.val_path]
    word2id = {'_PAD_': 0}

    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                out_list = []
                # 去除停用词
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in stopwords:
                        rt = re.findall('[a-zA-Z]+', word)
                        if word != '\t':
                            if len(rt) == 1:
                                continue
                            else:
                                out_list.append(word)
                for word in out_list:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """

    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs



if __name__ == "__main__":
    build_word2id('./word2vec_data/word2id.txt')
    splist = []
    word2id = {}
    with open('./word2vec_data/word2id.txt', encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()  # 去掉\n \t 等
            splist.append(sp)
        word2id = dict(splist)  # 转成字典
    for key in word2id:  # 将字典的值，从str转成int
        word2id[key] = int(word2id[key])

    id2word = {}  # 得到id2word
    for key, val in word2id.items():
        id2word[val] = key
    # 建立word2vec
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, Config.corpus_word2vec_path)











