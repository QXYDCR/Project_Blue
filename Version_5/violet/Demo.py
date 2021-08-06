import numpy as np

from sklearn.metrics import accuracy_score

# y_p = np.random.random((5, 2))
# print(y_p, y_p.shape)
#
# y_p = np.argmax(y_p, 1)
# print(y_p)
# y = np.random.randint(0, 2, 5)
# print(y)
#
# acc = accuracy_score(y, y_p)
# print(acc)



# 显示加载进度条
from tqdm import tqdm
#
# bar = tqdm(["a", "b", "c", "d"])
# for char in bar:
#     bar.set_description("Processing %s" % char)
#
# # Processing 9999: 100%|██████████| 9999/9999 [00:03<00:00, 3308.24it/s]
# idx = tqdm(np.arange(1, 10000))
# for i in idx:
#     idx.set_description("Processing %s" % i)



# ERROR:
# RuntimeError: 1D target tensor expected, multi-target not supported
import torch
import torch.nn.functional as F

# a = torch.Tensor([0, 1]).long()
# b = torch.Tensor([[0.8, 0.1], [0.9, 0.05]])
# # print(F.cross_entropy(input = b, target = a.unsqueeze(1)))
# # print(a.shape, b.shape)
#
#
# dd = torch.randint(0, 10, (1, 2, 3))
# print(dd, dd.shape)
#
# ee = dd.transpose(1, 0)
# print(ee, ee.shape)

import jieba
import nltk
from nltk import word_tokenize
s = "this is a string"
s1 = jieba.lcut(s)
print(s1)
s2 = nltk.pos_tag(s1)

nounwords = [name for name, value in s2 if value in ['NN', 'NNP']]





