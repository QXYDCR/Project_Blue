import pandas as pd
import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field,Example,Dataset
from torchtext.vocab import Vectors
from torchtext.legacy.data import Dataset,BucketIterator,Iterator, TabularDataset
from torch.nn import init
from tqdm import tqdm

"""
    提示：版本错误：在所有需要data的地方，用legacy.data代替
"""

SEED = 1234
torch.manual_seed(SEED)

# 1.数据
corpus = ["D'aww! He matches this background colour",
         "Yo bitch Ja Rule is more succesful then",
         "If you have a look back at the source"]
labels = [0,1,0]

# 2.定义不同的Field
TEXT = Field(sequential=True, lower=True, fix_length=10,tokenize=str.split,batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)
fields = [("comment", TEXT),("label",LABEL)]

# 3.将数据转换为Example对象的列表
examples = []
for text,label in zip(corpus,labels):
    example = Example.fromlist([text,label],fields=fields)
    examples.append(example)

print(type(examples[0]))
print(examples[0].comment)
print(examples[0].label)

# 4.构建词表
new_corpus = [example.comment for example in examples]
print(new_corpus)

TEXT.build_vocab(new_corpus)
# print(TEXT.process(new_corpus))

import jieba
# jieba分词返回的是迭代器，因此不能直接作为tokenize
print(jieba.cut("我爱北京天安门"))
# 使用lambda定义新的函数cut，其直接返回分词结果的列表，这样才可以作为tokenize
cut = lambda x:list(jieba.lcut(x))
cut("我爱北京天安门")

print(cut("我爱北京天安门"))

print(type(TEXT.vocab.freqs)) # freqs是一个Counter对象，包含了词表中单词的计数信息
print(TEXT.vocab.freqs['at'])
print(TEXT.vocab.itos[1]) # itos表示index to str
print(TEXT.vocab.stoi['<unk>']) # stoi表示str to index
print(TEXT.vocab.unk_index)
print(TEXT.vocab.vectors) # 词向量





















