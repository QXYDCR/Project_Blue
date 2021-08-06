from Util import *
from model import *

# load GloVe embeddings
# 40,000个
word2vectors, word2id = load_GloVe_twitter_emb()

# load the twitter dataset and splits in train/val/test
# 内容只有极性和内容，0为消极
#              polarity  tweet
# 8000         0         ill perform without her. Are you going tonight...
train, val, test = load_twitter_datasets(n_train=25000, n_val=8000)

# 修正句子 '@xoselena you got that right ' --> '<user> you got that right'
# tweets = train.tweet.apply(glove_preprocess).apply(normalize_text).values

# Create the tensors to feed to a Model
Xtrain, Ytrain = processAllTweets2vec(train, word2vectors)
Xval, Yval = processAllTweets2vec(val, word2vectors)
Xtest, Ytest = processAllTweets2vec(test, word2vectors)
# (25000, 50) (25000, 1)



# create Tensor datasets
train_data = TensorDataset(convertToTorchFloat(Xtrain), convertToTorchFloat(Ytrain))
val_data = TensorDataset(convertToTorchFloat(Xval), convertToTorchFloat(Yval))
test_data = TensorDataset(convertToTorchFloat(Xtest), convertToTorchFloat(Ytest))

# dataloaders
batch_size = 64

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)



mlp = NetMLP(input_size=50, layer_sizes=[128, 32], activation=nn.Tanh(),
          epochs=50, learning_rate=0.0002, l2reg=5e-4, dropout=0)
mlp = mlp.cuda() if torch.cuda.is_available() else mlp
print ("Number of parameters: {}".format(sum([np.prod(p.size()) for p in mlp.parameters()])))

history = mlp.fit(train_loader, val_loader)

test_performance = mlp.evaluate_loader(test_loader)
print ("\nTest  performance: loss={:.3f}, accuracy={:.3f}, precision={:.3f}, recall={:.3f}, f1={:.3f}".
       format(*[ test_performance[m] for m in ['loss', 'accuracy',
              'precision', 'recall', 'f1']]))


plot_perf(history, test_performance)

