import jieba
import torch

from Utils import *



# def update_critic(func, a, b):
#     return func(a, b)
#
#
# def update_actor(actions_pair, pred_actions, true_actions, H):
#     L_L_ = (true_actions == pred_actions.squeeze()).sum().item() / 30
#     R = H - 0.5 * L_L_
#     selected_logprobs = R * torch.gather(actions_pair, 1, pred_actions).squeeze()
#     loss_actor = selected_logprobs.mean()
#     return loss_actor
#
#
# def update_fc(func, a, b):
#     return func(a, b)


# I = np.random.rand(10, 3)
# print(I)
# print(uniform_score(I))
#
# G = uniform_score(I)
# print(np.sum(np.abs(G)))
# for i in range(10):
#     a = np.random.rand()
#     b = np.random.randint(0, 3)
#     print(a, b)

#
# x_input=torch.randn(10,3)#随机生成输入
# print('x_input:\n',x_input)
# y_target=torch.tensor([1, 2, 0, 1, 2, 0, 1, 2, 0, 1])#设置输出具体值 print('y_target\n',y_target)


# print(F.cross_entropy(x_input, y_target))
#
# a = torch.softmax(x_input, 1)
# a = torch.log(a)
# b = torch.gather(a, 1, y_target.unsqueeze(1)).squeeze()
# print(b.mean())

# a = '今天天气很好，但是我的心情很差!'
# b = load_stopwords()
#
#
# a = jieba.lcut(a)
# for w in a:
#     if w not in b:
#         print(w)


class Sentiment_Score():
    def __init__(self):
        super(Sentiment_Score, self).__init__()
        self.score_file = 'C:/Users/Chihiro/Desktop/RL Paper\Project\Project_Blue_1/' \
                          '数据集/Chinese_Corpus-master/sentiment_dict/sentiment_dict/sentiment_score.txt'
        self.degree_words_file = 'C:/Users/Chihiro/Desktop/RL Paper/Project' \
                                '/Project_Blue_1/数据集/情感极性词典/程度副词.txt'
        self.not_words_file = 'C:/Users/Chihiro/Desktop/RL Paper/Project/' \
                              'Project_Blue_1/数据集/情感极性词典/否定词.txt'

        # 词数：停用词:程度词:否定词 = 1426: 71: 71
        # 加载停用词，包含英文和中文
        self.stopwords = load_stopwords()
        # 加载Blonsp文件
        self.sentiwords = self.load_dict_scores(self.score_file)
        # 加载程度副词
        self.degree_words = load_dict(self.degree_words_file)
        # 加载否定词
        self.not_words = load_dict(self.not_words_file)

    def load_dict_scores(self, file):
        """
        :param file: (str), 词语分数文件
        :return: (dict), 词语分数字典
        """
        dict = {}
        with open(file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]
            for line in lines:
                line = line.strip().split(' ')
                try:
                    dict[line[0]] = np.float(line[1])
                except:
                    pass
        return dict


    def split_words(self, sentence, vocab):
        """
        :param sentence: (dict), {0: 'w', ...}
        :param vocab: (dict)
        :return:sen_word,{'w':0, ...}, unknown_word = [0, 1, ...]
        """
        sen_word = {}
        not_word = {}
        degree_word = {}
        unknown_word = []

        for idx in sentence.keys():
            w = sentence[idx]
            if w in self.sentiwords.keys() and w not in self.not_words.keys()\
                    and w not in self.degree_words.keys():
                sen_word[w] = idx
            elif w in self.not_words.keys() and w not in self.degree_words.keys():
                not_word[w] = idx
            elif w in self.degree_words.keys():    # 分词结果中在程度副词中的词
                degree_word[w] = idx
            else:
                unknown_word.append(idx)

        # 将分类结果返回
        return sen_word, not_word, degree_word, unknown_word

    def get_key_by_values(self, val, sent_word):
        key = [k for k, v in sent_word.items() if v == val]
        return key[-1]

    def get_sentence_scores(self, sentence, vocab):
        # 计算C值
        # 分解词
        # sen_word {'天气': 0, '好': 2, '心情': 7}
        sen_word, not_word, degree_word,unknown_word = self.split_words(sentence, vocab)
        # 用最后一个位置标明为情感词
        sen_word['_PAD_'] = len(sentence) - 1
        degree_flag = False

        ALPHA = 0.9
        # 权重初始化为1
        W = 1
        score = np.zeros(len(sentence))
        # 情感词下标初始化
        # 情感词的位置下标集合
        sent_idx = 0        # 迭代的情感词序号
        sen_word_idx = list(sen_word.values())    # 情感序号列表
        pre_idx = 0         # 前一个情感词的位置
        for i in range(len(sentence)):
            if i == 0:
                key = sentence[i]
                score[i] = self.sentiwords.get(key, 0)
                continue
            if i in sen_word.values():      # 如果序号在情感词中
                for j in range(pre_idx, i): # 搜寻前面的否定词
                    if j in not_word.values():
                        key = sentence[i]
                        W *= self.sentiwords.get(key, -1)
                    elif j in degree_word.values():
                        # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
                        key = self.get_key_by_values(j, degree_word)
                        W *= self.sentiwords.get(key, 1)
                        degree_flag = True
                key = sentence[i]
                # 得到当前情感词的评分
                if i == len(sentence) - 1:
                    if degree_flag == True:
                        score[i] = W
                        degree_flag = False
                    else:
                        score[i] = self.sentiwords.get(key, 0)
                else:
                    score[i] = self.sentiwords.get(key, 1) * W
                # 情感词下标加1，获取下一个情感词的位置

                if sent_idx < len(sentence):
                    pre_idx = sen_word_idx[sent_idx]
                sent_idx += 1
            else:
                key = sentence[i]
                score[i] = self.sentiwords.get(key, 0)

        return score, sen_word_idx

    def regular_text(self, sentence):
        """
        :param sentence: (list), ['w', 'b',...}
        :return: (dict), {1:'w'}
        """
        re_text = {}
        i = 0
        for w in sentence:
            re_text[i] = w
            i += 1
        return re_text

# a = '天气很好风和日丽但是我的心情不好'
# a = '我的心情很好虽然天气很差'
# a = '这是什么垃圾电影'
a = '虽然天气很差，但是我的心情很好'
# ------------------------ 2、读取数据 ------------------------
# 读取数据
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'
pos_sentences, neg_sentences,  vocab = read_data_from_pos_neg(good_file, bad_file)

d = jieba.lcut(a)

S = Sentiment_Score()
# ddd = []
# for i in d:
#     if i not in S.stopwords:
#         ddd.append(i)
# 
# content = ddd + ['_PAD_'] * (30 - len(ddd))
# eee = {}
# for i in range(len(content)):
#     eee[i] = content[i]
# print(eee)
# idf = {}
# for i, w in eee.items():
#     idf[w] = i
# mark_star()
# s = 0
# scores1 = 0
# for w in idf:
#     s = S.sentiwords.get(w, 0)
#     if s > 0:
#         s *= 0.4
#     scores1 += s
# print("极性1: ", scores1)
# 
# mark_star()
# g = S.get_sentence_scores(eee, vocab)
# print(g)
# s = 0
# scores2 = 0
# for w in g:
#     if w > 0:
#         w *= 0.4
#     scores2 += w
# print("极性2: ", scores2)
















