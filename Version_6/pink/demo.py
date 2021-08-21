import jieba
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
import string
from Util import *
# #
# from torch.utils.data import DataLoader, TensorDataset
# # from WS_model import *
# #
# #
# # # 去除随机性
# # SEED = 1234
# # torch.manual_seed(SEED)
# # np.random.seed(SEED)
# #
# # # -------------------------- 超参数 --------------------------
# # batch_size = 64
# #
# # # # -------------------------- 构建数据 --------------------------
# # # # 从Glove中加载词向量，并建立字典 size: [400000, 50]
# # # word2vectors, word2id = load_GloVe_twitter_emb()
# # #
# # # # 加载指定长度的训练集、测试集、验证集, 属于原始文本，未经处理, [25000, 2]
# # # train, val, test = load_twitter_datasets(n_train=2500, n_val=800)
# # # train_X = np.stack(train.tweet.apply(glove_preprocess).apply(
# # #         normalize_text))
# # # print(train_X[:5])
# # #
# # # # 停用词处理
# # # stop = stopwords.words("english") + list(string.punctuation)
# # # # 进行词性标注，生成词性标注列表
# # # ttt = [[i for i in word_tokenize(str(text).lower()) if i not in stop] for text in train_X[:1000]] #这里改数据量
# # # ttt = [nltk.pos_tag(t) for t in ttt]
# # # # 计数
# # # word_tag_fq = [nltk.FreqDist(t) for t in ttt]
# # # wordlist = [t.most_common() for t in word_tag_fq] # 合并计数
# # #
# # #
# # # # 进行词性归类
# # # df = []
# # # for wls in wordlist:
# # #     key = []
# # #     part = []
# # #     frequency = []
# # #     for i in range(len(wls)):
# # #         key.append(wls[i][0][0])
# # #         part.append(wls[i][0][1])
# # #         frequency.append(wls[i][1])
# # #     textdf = pd.DataFrame({
# # #         'key':key,
# # #         'part':part,
# # #         'frequency':frequency},columns=['key','part','frequency']
# # #     )
# # #     df.append(textdf)
# # #
# # # n = ['NN','NNP','NNPS','NNS','UH']
# # # v = ['VB','VBD','VBG','VBN','VBP','VBZ']
# # # a = ['JJ','JJR','JJS']
# # # r = ['RB','RBR','RBS','RP','WRB']
# # # for textdf in df:
# # #     for i in range(len(textdf['key'])):
# # #         z = textdf.iloc[i,1]
# # #         if z in n:
# # #             textdf.iloc[i,1]='n'
# # #         elif z in v:
# # #             textdf.iloc[i,1]='v'
# # #         elif z in a:
# # #             textdf.iloc[i,1]='a'
# # #         elif z in r:
# # #             textdf.iloc[i,1]='r'
# # #         else:
# # #             textdf.iloc[i,1]=''
# # #
# # # # 单词情感得分
# # # last_df = []
# # # for textdf in df:
# # #     score = []
# # #     for i in range(len(textdf['key'])):
# # #         m = list(swn.senti_synsets(textdf.iloc[i,0],textdf.iloc[i,1]))
# # #         s = 0
# # #         ra = 0
# # #         if len(m) > 0:
# # #             for j in range(len(m)):
# # #                 s += (m[j].pos_score()-m[j].neg_score())/(j+1)
# # #                 ra += 1/(j+1)
# # #             score.append(s/ra)
# # #         else:
# # #             score.append(0)
# # #     textdf = pd.concat([textdf,pd.DataFrame({'score':score})],axis=1) # 其实是创建副本，没有存进去,所以创建新列表
# # #     last_df.append(textdf)
# # #     print(textdf) # 打印每个句子的单词得分
# # #
# # #
# # # # 句子得分计算
# # # scorelis = [sum(last_df[i]['score']) for i in range(len(last_df))]
# # # data = pd.DataFrame(enumerate(scorelis),columns=['index','score_sentence'])
# # #
# # #
# # # data.to_csv('./HHH.csv')
# #
# # b = ['<user> i know u will come up w something great <elong> listening to a few of ur songs again ur voice is so beautiful <elong> yes even w o lyrics'
# #  'time to catch me some zz <elong> goodnight'
# #  '<user> you do know that <user> is within his rights to hassle you about going to the hospital to get it checked']
# #
# #
# #
# # # 801    <user> i know u will come up w something great...
# # # 801           time to catch me some zz <elong> goodnight
# # # 802    <user> you do know that <user> is within his r...
# # # 加载指定长度的训练集、测试集、验证集, 属于原始文本，未经处理, [25000, 2]
# # # 从Glove中加载词向量，并建立字典 size: [400000, 50]
# # # word2vectors, word2id = load_GloVe_twitter_emb()
# #
# #
# # train, val, test = load_twitter_datasets(n_train=2500, n_val=800)
# # train_X = np.stack(train.tweet.apply(glove_preprocess).apply(normalize_text))
# #
# #
# # # 从推特训练集中中，构建字典, 数据集在里面经过了清洗数据
# # # vocab = extractVocabulary(train)
# #
# # # 从训练集 & Glove词中找到交集，将每个词转化为w2id, word to glove vec
# # # restrictedWord2id, embMatrix, id2restrictedWord = vocabEmbeddings(vocab, word2vectors)
# #
# # # # 清洗数据，固定句长，便于batch处理,[25000, 40]
# # # Xtrain, Ytrain = processAllTweets2tok(train, restrictedWord2id)
# # # # # train_X = train_X[:3]
# # # Xtrain = Xtrain
# # print(train_X[:2])
# #
# # ttt = []
# # # 切分数据
# # for line in train_X[:2]:
# #     ttt.append(line.split(" "))
# # # 词性标注
# # ttt = [nltk.pos_tag(t) for t in ttt]
# #
# # # 计数 FreqDist({('<elong>', 'JJ'): 2
# # word_tag_fq = [nltk.FreqDist(t) for t in ttt]
# # # print(word_tag_fq)
# # wordlist = [t.most_common() for t in word_tag_fq] # 合并计数
# # # print(wordlist)
# #
# # # 进行词性归类
# # df = []
# # for wls in wordlist:
# #     key = []
# #     part = []
# #     frequency = []
# #     for i in range(len(wls)):
# #         key.append(wls[i][0][0])
# #         part.append(wls[i][0][1])
# #         frequency.append(wls[i][1])
# #     textdf = pd.DataFrame({
# #         'key':key,
# #         'part':part,
# #         'frequency':frequency},columns=['key','part','frequency']
# #     )
# #     df.append(textdf)
# # # print(df)
# # n = ['NN','NNP','NNPS','NNS','UH']
# # v = ['VB','VBD','VBG','VBN','VBP','VBZ']
# # a = ['JJ','JJR','JJS']
# # r = ['RB','RBR','RBS','RP','WRB']
# # for textdf in df:
# #     for i in range(len(textdf['key'])):
# #         z = textdf.iloc[i,1]
# #         if z in n:
# #             textdf.iloc[i,1]='n'
# #         elif z in v:
# #             textdf.iloc[i,1]='v'
# #         elif z in a:
# #             textdf.iloc[i,1]='a'
# #         elif z in r:
# #             textdf.iloc[i,1]='r'
# #         else:
# #             textdf.iloc[i,1]=''
# #
# # print(df)
# #
# # # 单词情感得分
# # last_df = []
# # for textdf in df:
# #     score = []
# #     b = len(textdf['key'])
# #     for i in range(len(textdf['key'])):
# #         gg = textdf
# #         ee, ff = textdf.iloc[i,0], textdf.iloc[i,1]
# #         m = list(swn.senti_synsets(textdf.iloc[i,0], textdf.iloc[i,1]))
# #         s = 0
# #         ra = 0
# #         if len(m) > 0:
# #             for j in range(len(m)):
# #                 s += (m[j].pos_score()-m[j].neg_score())/(j+1)
# #                 ra += 1/(j+1)
# #             score.append(s/ra)
# #         else:
# #             score.append(0)
# #     textdf = pd.concat([textdf, pd.DataFrame({'score':score})],axis=1) # 其实是创建副本，没有存进去,所以创建新列表
# #     last_df.append(textdf)
# #     # print(last_df) # 打印每个句子的单词得分
# #
# # print("* " * 10)
# # print(swn.senti_synsets('good', 'a')[0].pos_score())
# #
# #
# #
# #
# #

a = {'score':[] }

for i in range(5):
    a['score'] += i

print(a['score'])











