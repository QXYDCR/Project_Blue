from Utils import *
# from LSTM_model import LSTM_Model

"""
    最高精度
    test accuracy is 89.27%.
    train accuracy is 0.9375
"""

# 去除随机性
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# 读取文件内容并构建词典
vocab = read_build_vocab(['Dataset/train.txt'])
mark_star()

# 加载预定义词向量，并且将自定义词典中的单词映射成词向量
# 预加载的词向量为50维的wiki
word2vec_path = 'C:\\Users\\Chihiro\\Desktop\\RL Paper' \
                '\\Project\\Project_Blue_1\\词向量\\wiki_word2vec_50.bin'
word2vec = build_word2vec(word2vec_path, vocab)
mark_star()

# 将文本转化为数值形式
print('train corpus load: ')
train_contents, train_labels = convert_txt_to_num('./Dataset/train.txt', vocab)
print('\nvalidation corpus load: ')
val_contents, val_labels = convert_txt_to_num('./Dataset/validation.txt', vocab)
print('\ntest corpus load: ')
test_contents, test_labels = convert_txt_to_num('./Dataset/test.txt', vocab)


# 混合训练集和验证集
contents = np.vstack([train_contents, val_contents])
labels = np.concatenate([train_labels, val_labels])


# 加载训练用的数据
train_dataset = TensorDataset(torch.from_numpy(contents), torch.LongTensor(labels))
train_dataloader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True, drop_last = True)


# vocab_size, embedding_dim, hidden_size,
#                 pretrained_embed, is_updata_w2c, n_class):
# 定义模型
class LSTM_Model(nn.Module):
    def __init__(self, vocab_size = len(vocab), embedding_dim=50, hidden_size=50,
                pretrained_embed = word2vec, is_updata_w2c = True, n_class = 2):
        super(LSTM_Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.pretrained_embed = pretrained_embed
        self.is_updata_w2c = is_updata_w2c
        self.n_class = n_class

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = is_updata_w2c

        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, x, h, c):
        # x: [32, 50] --> [32, 50, 50]
        x = self.embedding(x)
        x = x.transpose(0, 1)

        outputs, (h, c) = self.lstm(x, (h, c))
        # [32, 50, 50] --> [32, 50]
        outputs = outputs[-1]  # [batch_size, n_hidden]
        # [32, 50] --> [32, 2]
        outputs = self.linear(outputs)
        return outputs, h, c

    def init_H_C(self):
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        h = torch.zeros(1, 32, self.hidden_size)
        c = torch.zeros(1, 32, self.hidden_size)
        return h, c


model = LSTM_Model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


# train
save_path = 'Mv.ztl'
max_acc = 0

for epoch in range(1):
    accurates = []
    for i, data in enumerate(train_dataloader):
        x, y = data
        y = y.squeeze()
        # x: [32, 50, 50], y: [32], out: [32, 50]
        h, c = model.init_H_C()
        output, h, c = model(x, h, c)

        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, _, acc_rate = get_accurate(output, y)
        accurates.append(acc_rate)
        if i % 200 == 0:
            rs = 0
            for r in accurates:
                rs += r
            rs = rs / len(accurates)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, acc: '.format(
                epoch + 1, i * len(x), len(train_dataset),
                100. * i / len(train_dataloader), loss.item()), rs)

            if rs > max_acc:
                torch.save(model.state_dict(), save_path)
                print("sava parameters")
                print("acc: ", rs)
                # max_acc = rs


# 加载模型
model.load_state_dict(torch.load(save_path))

# 加载训练用的数据
test_dataset = TensorDataset(torch.from_numpy(test_contents), torch.LongTensor(test_labels))
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = True, drop_last = True)

print(test_predict(test_dataloader, model))























