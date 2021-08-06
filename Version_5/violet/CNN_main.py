from Util import *
from model import *

# load GloVe embeddings
word2vectors, word2id = load_GloVe_twitter_emb()
# load the twitter dataset and splits in train/val/test
train, val, test = load_twitter_datasets(n_train=25000, n_val=8000)

tweets = train.tweet.apply(glove_preprocess).apply(normalize_text).values


# look into distribution of length of tweets to determine optimal padding
import matplotlib.pyplot as plt

# 统计推特数据集中，文本的长度分布, 以(10, 20)为中心
# train.tweet.apply(glove_preprocess).apply(normalize_text).apply(lambda x: len(x.split(' '))).hist()
# plt.title('Distribution of length of tweets')
# plt.show()


vocab = extractVocabulary(train)
restrictedWord2id, embMatrix = vocabEmbeddings(vocab, word2vectors)

Xtrain, Ytrain = processAllTweets2tok(train, restrictedWord2id)
Xval, Yval = processAllTweets2tok(val, restrictedWord2id)
Xtest, Ytest = processAllTweets2tok(test, restrictedWord2id)


# create Tensor datasets
train_data = TensorDataset(convertToTorchInt(Xtrain), convertToTorchFloat(Ytrain))
val_data = TensorDataset(convertToTorchInt(Xval), convertToTorchFloat(Yval))
test_data = TensorDataset(convertToTorchInt(Xtest), convertToTorchFloat(Ytest))

# dataloaders
batch_size = 128

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


cnn = NetCNN(vocab_size=embMatrix.shape[0], embedding_matrix=embMatrix,
             filter_sizes=[1, 2, 3, 5, 10], num_filters=8, embed_size=embMatrix.shape[1],
             finetune_emb=False, epochs=150, learning_rate=2e-5, l2reg=3e-3, dropout=0.1)

cnn = cnn.cuda() if torch.cuda.is_available() else cnn
print (cnn)
print (sum([np.prod(p.size()) for p in cnn.parameters()])-np.prod(embMatrix.shape))

history = cnn.fit(train_loader, val_loader)
test_performance = cnn.evaluate_loader(test_loader)
print ("Test performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}".
       format(*[ test_performance[m] for m in
                 ['loss', 'accuracy', 'precision', 'recall', 'f1']]))

plot_perf(history, test_performance)









