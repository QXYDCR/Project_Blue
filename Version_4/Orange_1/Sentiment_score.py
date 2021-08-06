from Utils import *

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------ 1、超参数设置 ------------------------
# 数据文件
good_file = 'data/good.txt'
bad_file = 'data/bad.txt'

BATCH = 1

# ------------------------ 2、读取数据 ------------------------
# 读取数据
pos_sentences, neg_sentences,  vocab = read_data_from_pos_neg(good_file, bad_file)


# 加载pos, neg词典
pos_dict_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/' \
                'Project_Blue/Version_3/Emotion_Dict/dict/positive.txt'
neg_dict_file = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/Code 102/' \
                'Project_Blue/Version_3/Emotion_Dict/dict/negative.txt'


pos_dict_std = load_dict(pos_dict_file)
neg_dict_std = load_dict(neg_dict_file)

score_file = 'C:/Users/Chihiro/Desktop/RL Paper\Project\Project_Blue_1/' \
             '数据集/Chinese_Corpus-master/sentiment_dict/sentiment_dict/sentiment_score.txt'
def load_dict_scores(file):
    dict = {}
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        for line in lines:
            line = line.strip().split(' ')
            try:
                dict[line[0]] = np.float(line[1])
            except:
                pass
    return dict

word_socre = load_dict_scores(score_file)


# 加载预定义词向量，并且将自定义词典中的单词映射成词向量
# 预加载的词向量为50维的wiki
word2vec_path = 'C:\\Users\\Chihiro\\Desktop\\RL Paper' \
                '\\Project\\Project_Blue_1\\词向量\\wiki_word2vec_50.bin'
word2vec = build_word2vec(word2vec_path, vocab)
mark_star()

# ------------------------ 3、数值化文本 ------------------------
pos_dataset, pos_labels = convert_emotion_text_to_num(good_file, vocab, lnum=0, max_len=30)
neg_dataset, neg_labels = convert_emotion_text_to_num(bad_file, vocab, lnum=1, max_len=30)


# ------------------------ 4、建立数据集 ------------------------
# 将正向、负向集合融合，在随机打乱
dataset, labels = mix_dataset(pos_dataset, pos_labels, neg_dataset, neg_labels)
dataset, labels = random_data(dataset, labels)

# 切分数据集
test_size = len(dataset) // 10
train_data = dataset[5 * test_size :]
train_label = labels[5 * test_size :]
train_data = np.asarray(train_data)
train_label = np.asarray(train_label)


actor_loss = []
critic_loss = []
fc_loss = []
rights = []
reward = []

acc_num = 0
accs = []


from Demo import Sentiment_Score
s = Sentiment_Score()
def train_orgin_accumulate_word():
    acc_num = 0
    accs = []
    for epoch in range(4):
        for i, data in enumerate(zip(train_data, train_label)):
            # x, y --> [50, 1], [1,]
            x, y = data
            x = torch.LongTensor(x).view(-1, 1)
            y = torch.LongTensor(y)

            scores = 0
            text = convert_num_to_txt(x, vocab)
            for j in range(x.shape[0]):
                w = idx2word(x[j], vocab)
                s = word_socre.get(w, 0)
                # if s > 0:
                #     s *= 0.4
                scores += s

            if scores > 0:
                label = 0
            elif scores < 0:
                label = 1
            else:
                label = 2

            if label == y.item():
                acc_num += 1

            if i % 400 == 0:
                accs.append(acc_num / 400)
                print("acc: ", 1.0 * acc_num / 400)
                acc_num = 0


    plt.plot(range(len(accs)), accs, '+-r')
    plt.show()


def train_AD_accumulate_word():
    acc_num = 0
    accs = []
    for epoch in range(4):
        for i, data in enumerate(zip(train_data, train_label)):
            # x, y --> [50, 1], [1,]
            x, y = data
            text = convert_num_to_txt(x, vocab)
            x = torch.LongTensor(x).view(-1, 1)
            y = torch.LongTensor(y)
            scores = 0
            I = []

            for j in range(x.shape[0]):
                w = idx2word(x[j], vocab)
                s = word_socre.get(w, 0)
                if s > 0:
                    s = s * 0.4
                I.append(s)

            if scores > 0:
                label = 0
            elif scores < 0:
                label = 1
            else:
                label = 2

            if label == y.item():
                acc_num += 1

            if i % 400 == 0:
                accs.append(acc_num / 400)
                print("acc: ", 1.0 * acc_num / 400)
                acc_num = 0


    plt.plot(range(len(accs)), accs, '+-r')
    plt.show()



def train_self_accumulate_word():
    acc_num = 0
    accs = []
    s = Sentiment_Score()
    for epoch in range(4):
        for i, data in enumerate(zip(train_data, train_label)):
            # x, y --> [50, 1], [1,]
            x, y = data
            text = convert_num_to_txt(x, vocab)
            text = s.regular_text(text)

            scores, sen_word_idx = s.get_sentence_scores(text, vocab)
            # for j in range(x.shape[0]):
            #     w = idx2word(x[j], vocab)
            #     s = word_socre.get(w, 0)
            #     I.append(s)
            score_sen = 0
            for jj in range(len(scores)):
                if scores[jj] > 0:
                    scores[jj] = scores[jj] * 0.45
                score_sen += scores[jj]
            scores = score_sen
            if scores > 0:
                label = 0
            elif scores < 0:
                label = 1
            else:
                label = 2

            if label == y.item():
                acc_num += 1

            if i % 400 == 0:
                accs.append(acc_num * 1.0  / 400)
                print("acc self: ", 1.0 * acc_num / 400)
                acc_num = 0


    plt.plot(range(len(accs)), accs, '+-r')
    plt.show()



mark_star('-', '+')
train_self_accumulate_word()













