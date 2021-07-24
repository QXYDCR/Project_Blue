from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import tqdm
from Config import Config












if __name__ == "__main__":
    split = []
    word2id = []

    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
        word2id = dict(split)

    # 将字典的值，从str转成int
    for key in word2id:
        word2id[key] = int(word2id[key])

    # 得到id2word
    id2word = {}
    for key, val in word2id.items():
        id2word[val] = key

    train_array, train_lable, val_array, val_lable,\
    test_array, test_lable = prepare_data(word2id,train_path=Config.train_path,
                                          val_path=Config.val_path,test_path=Config.test_path,
                                          seq_lenth=Config.max_sen_len)


