import spacy
import torch
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 1234
torch.manual_seed(SEED)


all_data = pd.read_csv('train.csv', sep='\t')
test_data = pd.read_csv('test.csv', sep='\t')

print(all_data.head())

train, val = train_test_split(all_data, test_size=0.2)

spacy_en = spacy.load('en')

# create a tokenizer function
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Field
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)


# Dataset
train,val = data.TabularDataset.splits(
        path='', train='train.csv',validation='val.csv', format='csv',skip_header=True,
        fields=[('name',None),('location',None),('quote', TEXT), ('age', LABEL)])

test = data.TabularDataset('test.tsv', format='tsv',skip_header=True,
        fields=[('name',None),('location',None),('quote', TEXT), ('age', LABEL)])

print(train[:5])














