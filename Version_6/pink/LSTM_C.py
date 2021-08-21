from Util import *
from model import *
from torch.utils.data import DataLoader, TensorDataset

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# 从Glove中加载词向量，并建立字典 size: [400000, 50]
word2vectors, word2id = load_GloVe_twitter_emb()

# 加载指定长度的训练集、测试集、验证集, 属于原始文本，未经处理, [25000, 50]
train, val, test = load_twitter_datasets(n_train=25000, n_val=8000)

# 从推特训练集中中，构建字典, 数据集在里面经过了清洗数据, set:4510
vocab = extractVocabulary(train)

# 从训练集 & Glove词中找到交集，将每个词转化为w2id, word to glove vec
# size: 3319, [3319, 50]
restrictedWord2id, embMatrix, id2restrictedWord = vocabEmbeddings(vocab, word2vectors)


# 清洗数据，固定句长，便于batch处理,[2500, 40]
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


# 定义模型
lstm_model = LSTM_Attention(vocab_size=embMatrix.shape[0], embedding_matrix=embMatrix, batch= 64,
             embed_size=embMatrix.shape[1],layer_num=1, hidden_size=50, n_class= 2,
             finetune_emb=False, epochs=20, learning_rate=0.001, l2reg=3e-3, dropout=0.1)

lstm_model = lstm_model.cuda() if torch.cuda.is_available() else lstm_model
# print (cnn)
# print (sum([np.prod(p.size()) for p in cnn.parameters()])-np.prod(embMatrix.shape))

# 训练数据
history = lstm_model.train_data(train_loader, val_loader)

test_performance = lstm_model.evaluate_loader(test_loader)
print ("Test performance: loss={:.3f}, accuracy={:.3f}".
       format(*[test_performance[m] for m in ['loss', 'accuracy']]))

plot_perf(history, test_performance)











