from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from model import *
from Util import *
from tool import *

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------- 超参数 --------------------------
batch_size = 64

# -------------------------- 构建数据 --------------------------
# 从Glove中加载词向量，并建立字典 size: [400000, 50]
word2vectors, word2id = load_GloVe_twitter_emb()

# 加载指定长度的训练集、测试集、验证集, 属于原始文本，未经处理, [25000, 2]
train, val, test = load_twitter_datasets(n_train=25000, n_val=8000)

# 从推特训练集中中，构建字典, 数据集在里面经过了清洗数据
vocab = extractVocabulary(train)

# 从训练集 & Glove词中找到交集，将每个词转化为w2id, word to glove vec
restrictedWord2id, embMatrix, id2restrictedWord = vocabEmbeddings(vocab, word2vectors)


# 清洗数据，固定句长，便于batch处理,[25000, 40]
Xtrain, Ytrain = processAllTweets2tok(train, restrictedWord2id)
Xval, Yval = processAllTweets2tok(val, restrictedWord2id)
Xtest, Ytest = processAllTweets2tok(test, restrictedWord2id)


# -------------------------- 转载数据 --------------------------
# 将数据集类型转换, numpy --> tensor, 大小为[train_data.shape[0], 40], 其中每个元素是id
train_data = TensorDataset(convertToTorchInt(Xtrain), convertToTorchInt(Ytrain))
val_data = TensorDataset(convertToTorchInt(Xval), convertToTorchInt(Yval))
test_data = TensorDataset(convertToTorchInt(Xtest), convertToTorchInt(Ytest))

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

model = Transformer(n_vocab=len(vocab), embed_matrix= embMatrix)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)



def compute_loss(x, y):

    predictions = model(x)
    # y, [64, 1] --> [64,]
    loss = loss_func(predictions, y.squeeze())
    outputs = torch.argmax(predictions, 1)
    accuracy = accuracy_score(y.squeeze(), outputs)

    return loss, accuracy

def evaluate_loader(loader):
    # compute loss and accuracy for that loader
    metrics = {'loss': 0, 'accuracy': 0, }
    # loop over examples of loader
    for i, (x, y) in enumerate(loader):
        loss, accuracy = compute_loss(x, y)
        # sum up metrics in dict
        metrics['loss'] += loss.item()
        metrics['accuracy'] += accuracy
    # normalize all values
    for k in metrics.keys():
        metrics[k] /= len(loader)

    return metrics

history = {'loss': [], 'val_loss': [],'accuracy': [], 'val_accuracy': [],}

for epoch in range(10):
    # one epoch
    train_metrics = {'loss': 0, 'accuracy': 0, }
    for i, (x, y) in enumerate(train_loader):
        # Forward + Backward + Optimize
        loss, accuracy = compute_loss(x, y)
        train_metrics['loss'] += loss.item()
        train_metrics['accuracy'] += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 归一化分数
    for k in train_metrics.keys():
        train_metrics[k] /= len(train_loader)

    # 计算验证集的准确性
    val_metrics = evaluate_loader(val_loader)

    # save metrics in history
    for key in train_metrics:
        history[key].append(train_metrics[key])
    for key in val_metrics:
        history['val_' + key].append(val_metrics[key])

        # printing of performance at freq_prints frequency
        if (epoch + 1) % 1 == 0:
            print("Epoch {}/{}\nTrain performance: loss={:.3f}, accuracy={:.3f}".format(
                epoch + 1, epoch, history['loss'][-1], history['accuracy'][-1]))


test_performance = evaluate_loader(test_loader)
print ("Test performance: loss={:.3f}, accuracy={:.3f}".
       format(*[test_performance[m] for m in ['loss', 'accuracy']]))

plot_perf(history, test_performance)



