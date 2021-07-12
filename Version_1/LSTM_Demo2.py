import numpy as np
import torch
import torch.nn as nn


# define the data
seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'haku', 'star']

# build the dictionary
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']

word_to_idx = {w: i for i, w in enumerate(char_arr)}
idx_to_word = {i: w for i, w in enumerate(char_arr)}

# define the class
n_class = len(set(word_to_idx))
seq_len = len(seq_data)

# create data
def make_data(sentences = seq_data, is_tonsor = True):
    xs = []
    ys = []
    for sen in sentences:
        temp = [word_to_idx[w] for w in sen[:-1]]
        xs.append(np.eye(n_class)[temp])
        ys.append(word_to_idx[sen[-1]])

    if is_tonsor == True:
        xs = torch.FloatTensor(xs)
        ys = torch.LongTensor(ys)
    return xs, ys



# seq : 3, x_feature: 26, hidden; x, batch = 26
input_size = 26
hidden_size = 128
class LSTM_1(nn.Module):
    def __init__(self):
        super(LSTM_1, self).__init__()
        self.LSTM = nn.LSTM(input_size=n_class, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, n_class)

    def forward(self, h_0, x, c_0):
        # x: [b, S_L, f_x] --> [S_L, b,f_x]
        x = x.transpose(0, 1)
        # out: [S_L, b,f_x] --> [S_L, b, hidden]
        out, (h_0, c_0) = self.LSTM(x, (h_0, c_0))
        # [S_L, b, hidden] --> [b, hidden]
        out = out[-1]
        # [b, hidden] --> [b, n_class]
        out = self.fc(out)
        return out


LSTM = LSTM_1()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LSTM.parameters(), lr = 0.01)

batch = 1
h_0 = torch.zeros([1, 10, hidden_size])
c_0 = torch.zeros([1, 10, hidden_size])

def get_accurate(predict, y):
    LSTM.eval()

    pre_y = torch.max(predict, dim=1)[1].numpy()
    y = y.data.numpy()
    print(type(pre_y), type(y))
    acc = (pre_y == y).sum()
    print(pre_y == y)
    return acc * 1.0 / len(y)

for epoch in range(1000):
    # xs:, tensor[10, 3, 26], ys[10, 1]
    xs, ys = make_data()

    predict_ys = LSTM(h_0, xs, c_0)

    loss = loss_func( predict_ys, ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss))
        print("acc: ",get_accurate(predict_ys, ys))









