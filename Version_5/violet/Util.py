import tqdm
import numpy as np
import pandas as pd
import re
from string import punctuation
from nltk.stem.porter import *
import torch
import matplotlib.pyplot as plt
# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

glove_path = 'C:/Users/Chihiro/Desktop/RL Paper/Project/Project_Blue_1/词向量/glove.6B.50d.txt'
def load_GloVe_twitter_emb(path_glove = glove_path):
    '''
        Loading GloVe pretraiend embeddings and storing them in two dictionaries,
        one maps words to embedding vectors of size 200, the other maps words to ids.

        inputs:
            - path_glove (str):      path where the GloVe Twitter text file is stored
        return:
            - word2vectors (dict):   words mapped to GloVe vectors
            - word2id (dict):        words mapped to id
        '''
    word2vectors, word2id = {}, {}
    count = 0  # counter for word ids

    with open(f'{path_glove}', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            # word2vectors dictionary with words to GloVe embedding vectors
            word2vectors[word] = np.array(line[1:]).astype(np.float)
            # word2id dictionary with words to id
            word2id[word] = count
            count += 1
    return word2vectors, word2id


path_data_train_str = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code/' \
'day4_1GloVe_Sentiment_Analysis_MLP_CNN_TwitterSentiment140-master/training.1600000.processed.noemoticon.csv'

path_data_test_str = 'C:/Users/Chihiro/Desktop/RL Paper/Git Code' \
'/day4_1GloVe_Sentiment_Analysis_MLP_CNN_TwitterSentiment140-master/testdata.manual.2009.06.14.csv'


def load_twitter_datasets(path_data_train=path_data_train_str,
                          path_data_test=path_data_test_str,
                          n_train=25000, n_val=8000):
    '''
        Loading twitter data from train and test file. Splitting train set to train and validation set.
        Default size 25000 of train samples and 8000 of validation samples.
        Tweets with polarity of neutral label are disregarded; polarity labels are converted to binary
        representation (positive sentiment labeled 1, negative sentiment labeled 0).

        inputs:
            - path_data_train (str):  path to train data (csv)
            - path_data_test (str):   path to test data (csv)
            - n_train (int):          number of training samples kept
            - n_val (int):            number of validation samples kept
        return:
            - train (pd.dataFrame):   dataframe with polarity (label) and tweet (sentence) for train
            - val (pd.dataFrame):     dataframe with polarity (label) and tweet (sentence) for validation
            - test (pd.dataFrame):    dataframe with polarity (label) and tweet (sentence) for test
    '''
    # TRAIN and VAL DATA -- loading the train data
    train = pd.read_csv(path_data_train, encoding='latin-1', header=0,
                        names=["polarity", "id", "date", "query", "user", "tweet"])

    # original polarity column has values: {0, 2, 4} = {negative, neutral, positive}
    # drop neutral labels in polarity column and divide by 4 to make labels binary
    train = train[train.polarity != 2]
    # 将极性[0, 4], 变为{0， 1] == [neg, pos]
    train.polarity = train.polarity // 4

    # droppings all columns but polarity score and the tweet
    train = train[["polarity", "tweet"]]
    # shuffling the rows to obtain val and train subsets
    train = train.sample(frac=1).reset_index(drop=True)
    val = train.iloc[:n_val]
    train = train.iloc[n_val:n_val + n_train]

    # TEST DATA -- loading the test data
    test = pd.read_csv(path_data_test, encoding='latin-1', header=0,
                       names=["polarity", "id", "date", "query", "user", "tweet"])

    # drop neutral labels in polarity column and divide by 4 to make labels binary
    test = test[test.polarity != 2]
    test.polarity = test.polarity // 4
    test = test[["polarity", "tweet"]]

    return train, val, test


def hashtags_preprocess(x):
    '''
        Creating a hashtag token and processing the formatting of hastags, i.e. separate uppercase words
        if possible, all letters to lowercase.


        inputs:
            - x (regex group):        x.group(1) contains the text associated with a hashtag

        return:
            - text (str):             preprocessed text
    '''
    s = x.group(1)
    if s.upper() == s:
        # if all text is uppercase, then tag it with <allcaps>
        return ' <hashtag> ' + s.lower() + ' <allcaps> '
    else:
        # else attempts to split words if uppercase starting words (ThisIsMyDay -> 'this is my day')
        return ' <hashtag> ' + ' '.join(re.findall('[A-Z]*[^A-Z]*', s)[:-1]).lower()


def allcaps_preprocess(x):
    '''
        If text/word written in uppercase, change to lowercase and tag with <allcaps>.
        inputs:
            - x (regex group):        x.group() contains the text

        return:
            - text (str):             preprocessed text
    '''
    return x.group().lower() + ' <allcaps> '


def glove_preprocess(text):
    '''
        To be consistent with use of GloVe vectors, we replicate most of their preprocessing.
        Therefore the word distribution should be close to the one used to train the embeddings.
        Adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

        inputs:
            - text (str):    tweet to be processed

        return:
            - text (str):    preprocessed tweet
    '''
    # for tagging urls
    text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/|www\.){1}[A-Za-z0-9.\/\\]+[]*', ' <url> ', text)
    # for tagging users
    text = re.sub("\[\[User(.*)\|", ' <user> ', text)
    text = re.sub('@[^\s]+', ' <user> ', text)
    # for tagging numbers
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
    # for tagging emojis
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("<3", ' <heart> ', text)
    text = re.sub(eyes + nose + "[Dd)]", ' <smile> ', text)
    text = re.sub("[(d]" + nose + eyes, ' <smile> ', text)
    text = re.sub(eyes + nose + "p", ' <lolface> ', text)
    text = re.sub(eyes + nose + "\(", ' <sadface> ', text)
    text = re.sub("\)" + nose + eyes, ' <sadface> ', text)
    text = re.sub(eyes + nose + "[/|l*]", ' <neutralface> ', text)
    # split / from words
    text = re.sub("/", " / ", text)
    # remove punctuation
    text = re.sub('[.?!:;,()*]+', ' ', text)
    # tag and process hashtags
    text = re.sub(r'#([^\s]+)', hashtags_preprocess, text)
    # for tagging allcaps words
    text = re.sub("([^a-z0-9()<>' `\-]){2,}", allcaps_preprocess, text)
    # find elongations in words ('hellooooo' -> 'hello <elong>')
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <elong> ", text)
    return text


# constants needed for normalize text functions
non_alphas = re.compile(u'[^A-Za-z<>]+')
cont_patterns = [
    ('(W|w)on\'t', 'will not'),
    ('(C|c)an\'t', 'can not'),
    ('(I|i)\'m', 'i am'),
    ('(A|a)in\'t', 'is not'),
    ('(\w+)\'ll', '\g<1> will'),
    ('(\w+)n\'t', '\g<1> not'),
    ('(\w+)\'ve', '\g<1> have'),
    ('(\w+)\'s', '\g<1> is'),
    ('(\w+)\'re', '\g<1> are'),
    ('(\w+)\'d', '\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


def normalize_text(text):
    '''
        Final cleanup of text by removing non-alpha characters like '\n', '\t'... and
        non-latin characters + stripping.
        inputs:
            - text (str):    tweet to be processed
        return:
            - text (str):    preprocessed tweet
    '''
    clean = text.lower()
    clean = clean.replace('\n', ' ')
    clean = clean.replace('\t', ' ')
    clean = clean.replace('\b', ' ')
    clean = clean.replace('\r', ' ')
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return u' '.join([y for y in non_alphas.sub(' ', clean).strip().split(' ')])


# nltk stemmer
# 英文分词工具, 将一类词转化为，相近的一个词
stemmer = PorterStemmer()


def tweet2Vec(tweet, word2vectors):
    '''
        Takes in a processed tweet, tokenizes it, converts to GloVe embeddings
        (or zeroes if words are unknown) and applies average pool to obtain one vector for that tweet.
        inputs:
            - tweet (str):             one raw tweet from the dataset
            - word2vectors (dict):     GloVe words mapped to GloVe vectors
        return:
            - embeddings (np.array):   resulting sentence vector (shape: (200,))
    '''
    return np.mean([word2vectors.get(stemmer.stem(t), np.zeros(shape=(50, )))
                    for t in tweet.split(" ")], 0)



def processAllTweets2vec(df, word2vectors):
    '''
        # 将所有的文本转化为规则的文本
        Takes in dataframe of labels and tweets and applies preprocessing on all tweets
        (glove_preprocess -> normalize_text -> tweet2Vec) to build X matrix and create
        Y matrix of labels.
        inputs:
            - df (pd.dataFrame):       dataframe of polarity and tweets
            - word2vectors (dict):     GloVe words mapped to GloVe vectors
        return:
            - X (np.array):            vector of tweets (shape: (df.shape[0], 50))
            - Y (np.array):            vector of labels (shape: (df.shape[0], 1))
    '''
    X = np.stack(df.tweet.apply(glove_preprocess).apply(
        normalize_text).apply(lambda x: tweet2Vec(x, word2vectors)))
    Y = df.polarity.values.reshape(-1, 1)
    return X, Y


def convertToTorchFloat(x):
    '''
        Converting np.array to torch.tensor (float) and, if availabe, convert to cuda.
        inputs:
            - x (np.array):            tensor
        return:
            - tensor (torch.tensor):   converted float tensor
    '''
    x = torch.from_numpy(x).float()
    return x.cuda() if torch.cuda.is_available() else x


def convertToTorchInt(x):
    '''
        Converting np.array to torch.tensor (int64) and, if availabe, convert to cuda.


        inputs:
            - x (np.array):            tensor

        return:
            - tensor (torch.tensor):   converted int64 tensor
    '''
    x = torch.from_numpy(x).to(torch.int64)
    return x.cuda() if torch.cuda.is_available() else x



def plot_perf(history, final_perf):
    '''
        Function to plot performance plots, i.e. evolution of metrics during training
        on train, val and test sets.
    '''
    epochs = range(1, len(history['loss'])+1)
    for key in ['loss', 'accuracy']:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history[key], '+-b', label=key)
        plt.plot(epochs, history['val_'+key], '+-g', label='val_'+key)
        plt.axhline(y=final_perf[key], color='r', linestyle='--', label='test_'+key)
        plt.legend()
        plt.title('Evolution of {} during training'.format(key))
        plt.plot()

    plt.show()


def vocabEmbeddings(vocab, word2vectors):
    '''
        Given a set of unique words (vocabulary), a mapping from word to unique id
        and a mapping from word to vector (GloVe) we build a restricted vocabulary
        and an embedding matrix
        inputs:
            - vocab (set):                 set of unique words in vocabulary of training set
            - word2vectors (dict):         original GloVe words mapped to GloVe vectors

        return:
            - restrictedWord2id (dict):    vector of tweets (shape: (df.shape[0], 50))
            - embMatrix (np.array):        embedding matrix of shape (len(restrictedWord2id), 50)
    '''
    # get intersection of both vocabularies (training and glove)
    # 原本词典17,000左右 & 40,0000 = 10674
    keys = word2vectors.keys() & vocab
    # 字典形式, 大小为10674
    restrictedWord2id = dict(zip(keys, range(len(keys))))
    id2restrictedWord = {v: k for k, v in restrictedWord2id.items()}

    # create embedding matrix from the vocab，(10617, 50)
    embMatrix = np.array([word2vectors[id2restrictedWord[idx]] for idx in range(len(id2restrictedWord))])
    # add unknown token -> average of all tokens from vocab (as suggested by Pennington here:
    # https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
    # (10618, 50)
    embMatrix = np.vstack((embMatrix, embMatrix.mean(0)))
    print(embMatrix.shape)
    # add a padding token -> initialize to 0, # (10619, 50)
    embMatrix = np.vstack((embMatrix, np.zeros((1, embMatrix.shape[1]))))

    return restrictedWord2id, embMatrix, id2restrictedWord


def extractVocabulary(df):
    '''
        Creating a set of (unique) words, i.e. vocabulary from tweets.
        inputs:
            - df (pd.dataFrame):    dataFrame with tweets to extract vocabulary from
        return:
            - vocab (set):          unique words in vocabulary
    '''
    vocab = set()
    # get an array with all tweets
    tweets = df.tweet.values
    # tqdn, 显示加载进度条
    for t in tqdm.tqdm(tweets):
        words = normalize_text(glove_preprocess(t)).split(' ')
        # applying stemming to each word
        for w in words:
            vocab.add(stemmer.stem(w))
    return vocab

def get_words_from_num(line, id2word):
    text = []
    for i in line:
        text.append(id2word.get(i, 'PAD'))
    return text


def tweet2tok(tweet, word2id, pad_length = 40):
    '''
        Split to tokens a processed tweet and create vector of glove vectors
        (or unknown tokens if OOV) and pad with pad tokens.
        inputs:
            - tweet (str):                 processed tweet
            - word2id (dict):              GloVe words mapped to GloVe ids (as in embedding matrix)
            - pad_length (int):            max sequence length
        return:
            - restrictedWord2id (dict):    vector of tweets (shape: (df.shape[0], 200))
            - embMatrix (np.array):        embedding matrix of shape (len(restrictedWord2id), 200)
    '''
    tweets = tweet.split(" ")
    # since we add unknown token in embedding matrix, its id is the len of vocab,
    # and same for padding token which is len(vocab)+1
    # 在标准词向量中，我们额外添加了两行，一行用于处理unknown，一行用于填充字符
    return np.array([word2id.get(stemmer.stem(t), len(word2id)) for
                     t in tweets[:min(pad_length, len(tweets))]] + max(
        pad_length - len(tweets), 0) * [len(word2id) + 1])


def processAllTweets2tok(df, word2id, pad_length=40):
    '''
        Takes in dataframe of labels and tweets and applies preprocessing on all tweets
        (glove_preprocess -> normalize_text -> tweet2tok) to build X matrix (sequential) and create
        Y matrix of labels. Use padding as provided.
        inputs:
            - df (pd.dataFrame):       dataframe of polarity and tweets
            - word2id (dict):          GloVe words mapped to GloVe ids (as in embedding matrix)
        return:
            - X (np.array):            vector of tweets (shape: (df.shape[0], pad_length, 200))
            - Y (np.array):            vector of labels (shape: (df.shape[0], 1))
    '''
    X = np.stack(df.tweet.apply(glove_preprocess).apply(
        normalize_text).apply(lambda x: tweet2tok(x, word2id, pad_length)))
    Y = df.polarity.values.reshape(-1, 1)
    return X, Y








class SentimentScore():
    """
        用于计算句子或者单词的得分
    """
    def __init__(self):
        super(SentimentScore, self).__init__()

    def split_line(self, line):
        cols = line.split("\t")
        return cols

    def get_words(self, cols):
        words_ids = cols[4].split(" ")
        words = [w.split("#")[0] for w in words_ids]
        return words

    def get_positive(self, cols):
        return cols[2]

    def get_negative(self, cols):
        return cols[3]

    def get_objective(self, cols):
        return 1 - (float(cols[2]) + float(cols[3]))

    def get_gloss(self, cols):
        return cols[5]

    def get_scores(self, filepath, sentiword):
        f = open(filepath)
        totalobject = 0.0
        count = 0.0
        totalpositive = 0.0
        totalnegative = 0.0
        for line in f:
            if not line.startswith("#"):
                cols = self.split_line(line)
                words = self.get_words(cols)
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
                            totalobject = totalobject + self.get_objective(cols)
                            totalpositive = totalpositive + float(self.get_positive(cols))
                            totalnegative = totalnegative + float(self.get_negative(cols))
                            count = count + 1
        score = 0
        if count > 0:
            if totalpositive > totalnegative:
                score = totalpositive / count
            else:
                score = -totalnegative / count

        return score













