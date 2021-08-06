import re
from nltk.corpus import stopwords

def split_line(line):
    cols = line.split("\t")
    return cols

def get_words(cols):
    words_ids = cols[4].split(" ")
    words = [w.split("#")[0] for w in words_ids]
    return words

def get_positive(cols):
    return cols[2]

def get_negative(cols):
    return cols[3]

def get_objective(cols):
    return 1 - (float(cols[2]) + float(cols[3]))

def get_gloss(cols):
    return cols[5]


def get_scores(filepath, sentiword):
    f = open(filepath)
    totalobject = 0.0
    count = 0.0
    totalpositive = 0.0
    totalnegative = 0.0
    for line in f:
        if not line.startswith("#"):
            cols = split_line(line)
            words = get_words(cols)
            # print(words)

            for word in sentiword:
                if word in words:
                    if word == "not":
                        totalobject = totalobject + 0
                        totalpositive = totalpositive + 0
                        totalnegative = totalnegative + 16
                        count = count + 1
                    else:
                        # print("For given word {0} - {1}".format(word, get_gloss(cols)))
                        # print("P Score: {0}".format(get_positive(cols)))
                        # print("N Score: {0}".format(get_negative(cols)))
                        # print("O Score: {0}\n".format(get_objective(cols)))
                        totalobject = totalobject + get_objective(cols)
                        totalpositive = totalpositive + float(get_positive(cols))
                        totalnegative = totalnegative + float(get_negative(cols))
                        count = count + 1

    if totalpositive > totalnegative:
        return 1
    return 0

    # if count > 0:
    #     if totalpositive > totalnegative:
    #         print("Positive word : 1")
    #         print("Positive value : ", totalpositive)
    #         print("Negative value : ", totalnegative)
    #     else:
    #         print("Negative : -1")
    #         print("Positive value : ", totalpositive)
    #         print("Negative value : ", totalnegative)
    #
    #     print("average object Score : ", totalobject / count)
    #     print("average totalpositive Score : ", totalpositive / count)
    #     print("average totalnegative Score : ", totalnegative / count)



from Util import *
from LSTM_model import *


# 从Glove中加载词向量，并建立字典 size: [400000, 50]
word2vectors, word2id = load_GloVe_twitter_emb()

# 加载指定长度的训练集、测试集、验证集, 属于原始文本，未经处理
train, val, test = load_twitter_datasets(n_train=25000, n_val=8000)

# 从推特训练集中中，构建字典, 数据集在里面经过了清洗数据
vocab = extractVocabulary(train)

# 从训练集 & Glove词中找到交集，将每个词转化为w2id, word to glove vec
restrictedWord2id, embMatrix, id2restrictedWord = vocabEmbeddings(vocab, word2vectors)


# 清洗数据，固定句长，便于batch处理
Xtrain, Ytrain = processAllTweets2tok(train, restrictedWord2id)
Xval, Yval = processAllTweets2tok(val, restrictedWord2id)
Xtest, Ytest = processAllTweets2tok(test, restrictedWord2id)

# 将数据集类型转换, numpy --> tensor, 大小为[train_data.shape[0], 40], 其中每个元素是id
train_data = TensorDataset(convertToTorchInt(Xtrain), convertToTorchInt(Ytrain))
val_data = TensorDataset(convertToTorchInt(Xval), convertToTorchInt(Yval))
test_data = TensorDataset(convertToTorchInt(Xtest), convertToTorchInt(Ytest))

batch_size = 64

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)


for epoch in range(4):
    acc_num = 0
    for i, data in enumerate(Xtrain, Ytrain):
        x, y = data
        text = get_words_from_num(x, id2restrictedWord)
        y_ = get_scores("SentiWordNet_3.0.0_20130122.txt", text)

        if y_ == y:
            acc_num += 1

    print("acc: ", acc_num / len(Xtrain))


# if __name__ == "__main__":
#     comment = input("Enter Your feeling : ")
#     sentiword = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", comment).split())
#     stop_words = set(stopwords.words('english'))
#
#     sentiword = sentiword.lower().split(" ")
#     filtered_sentence = [w for w in sentiword if not w in stop_words]
#     # print(filtered_sentence)
#     get_scores("SentiWordNet_3.0.0_20130122.txt", filtered_sentence)





