from Util import *
import nltk
from nltk.corpus import sentiwordnet as swn #得到单词情感得分



class MySentiWorld():
    """
        给定一句数值化文本，返回每个词对应的分数
    """
    def __init__(self, id2word):
        self.id2word = id2word

    def get_text(self, num):
        return num_to_seq(num, self.id2word)

    def get_pos_tag(self, num):
        text = self.get_text(num)
        return nltk.pos_tag(text)

    def get_scores(self, num):
        # 数值-->文本
        text = num_to_seq(num, self.id2word)
        # 添加情感标签
        pos_text = nltk.pos_tag(text)

        # 转化标签
        n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
        v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        a = ['JJ', 'JJR', 'JJS']
        r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']
        # [['poor', 'n'], ...]
        pos_dict = []
        for textdf in pos_text:
            temp = []
            w = textdf[0]
            temp.append(w)
            z = textdf[1]
            if z in n:
                temp.append('n')
            elif z in v:
                temp.append('v')
            elif z in a:
                temp.append('a')
            elif z in r:
                temp.append('r')
            else:
                temp.append('')
            pos_dict.append(temp)

        # 单词情感得分
        score = []

        for w_s_pair in pos_dict:
            w = w_s_pair[0]
            s = w_s_pair[1]
            m = list(swn.senti_synsets(w, s))
            s = 0
            ra = 0      # 权重参数，优先级越低，值越小
            if len(m) > 0:
                for j in range(len(m)):
                    s += (m[j].pos_score() - m[j].neg_score()) / (j + 1)
                    ra += 1 / (j + 1)
                score.append(s / ra)
            else:
                score.append(0)

        return score

    def get_batch_scores(self, x):
        # x: [b, input_size]
        batch_scores = []
        for i in range(x.shape[0]):
            y = x[i]
            score = self.get_scores(y)
            batch_scores.append(score)

        return np.array(batch_scores)

