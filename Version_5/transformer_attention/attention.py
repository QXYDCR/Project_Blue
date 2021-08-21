import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader

"""
Note:
    用于了解attention机制的使用
"""

zidian = {
    '<PAD>': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '0': 10,
    'Jan': 11,
    'Feb': 12,
    'Mar': 13,
    'Apr': 14,
    'May': 15,
    'Jun': 16,
    'Jul': 17,
    'Aug': 18,
    'Sep': 19,
    'Oct': 20,
    'Nov': 21,
    'Dec': 22,
    '-': 23,
    '/': 24,
    '<SOS>': 25,
    '<EOS>': 26,
}



class DateDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        # 随机生成一个日期
        date = np.random.randint(143835585, 2043835585)
        date = datetime.datetime.fromtimestamp(date)

        # 格式化成两种格式
        # 05-06-15
        # 15/Jun/2005
        date_cn = date.strftime("%y-%m-%d")
        date_en = date.strftime("%d/%b/%Y")

        # 中文的就是简单的拿字典编码就行了
        date_cn_code = [zidian[v] for v in date_cn]

        # 英文的,首先要在收尾加上标志位,然后用字典编码
        date_en_code = []
        date_en_code += [zidian['<SOS>']]
        date_en_code += [zidian[v] for v in date_en[:3]]
        date_en_code += [zidian[date_en[3:6]]]
        date_en_code += [zidian[v] for v in date_en[6:]]
        date_en_code += [zidian['<EOS>']]

        return torch.LongTensor(date_cn_code), torch.LongTensor(date_en_code)


dataloader = DataLoader(dataset=DateDataset(),
                        batch_size=100,
                        shuffle=True,
                        drop_last=True)

sample = 0
# 遍历数据
for i, data in enumerate(dataloader):
    sample = data
    break

print(sample[0][:5], sample[0].shape, sample[1][:5], sample[1].shape)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        #encoder
        #一共27个词,编码成16维向量
        self.encoder_embed = nn.Embedding(num_embeddings=27, embedding_dim=16)

        #输入是16维向量,隐藏层是32维向量
        self.encoder = nn.LSTM(input_size=16,
                               hidden_size=32,
                               num_layers=1,
                               batch_first=True)

        #decoder
        #一共27个词,编码成16维向量
        self.decoder_embed = nn.Embedding(num_embeddings=27, embedding_dim=16)

        #输入是16维向量,隐藏层是32维向量
        self.decoder_cell = nn.LSTMCell(input_size=16, hidden_size=32)

        #输入是64维向量,输出是27分类
        self.out_fc = nn.Linear(in_features=64, out_features=27)

        #注意力全连接层
        self.attn_fc = nn.Linear(in_features=32, out_features=32)

    def get_attn(self, out_x, h):

        #[b,32] -> [b,32]
        attn = self.attn_fc(h)

        #[b,32] -> [b,1,32]
        attn = attn.unsqueeze(dim=1)

        #[b,8,32] -> [b,32,8]
        out_x_T = out_x.permute(0, 2, 1)

        #[b,1,32],[b,32,8] -> [b,1,8]
        attn = torch.matmul(attn, out_x_T)

        #[b,1,8] -> [b,1,8]
        attn = F.softmax(attn, dim=2)

        #[b,1,8],[b,8,32] -> [b,1,32]
        attn = torch.matmul(attn, out_x)

        #[b,1,32] -> [b,32]
        attn = attn.squeeze()

        return attn

    def forward(self, x, y):
        #x编码
        #[b,8] -> [b,8,16]
        x = self.encoder_embed(x)

        #进入循环网络,得到记忆
        #[b,8,16] -> [b,8,32],[1,b,32],[1,b,32]
        out_x, (h, c) = self.encoder(x, None)

        #[1,b,32],[1,b,32] -> [b,32],[b,32]
        h = h.squeeze()
        c = c.squeeze()

        #丢弃y的最后一个词
        #因为训练的时候是以y的每一个词输入,预测下一个词
        #所以不需要最后一个词
        #[b,11] -> [b,10]
        y = y[:, :-1]

        #y编码
        #[b,10] -> [b,10,16]
        y = self.decoder_embed(y)

        #用cell遍历y的每一个词
        outs = []
        for i in range(10):

            attn = self.get_attn(out_x, h)

            #把y的每个词依次输入循环网络
            #第一个词的记忆是x的最后一个词的记忆
            #往后每个词的记忆是上一个词的记忆
            #[b,16] -> [b,32],[b,32]
            h, c = self.decoder_cell(y[:, i], (h, c))

            #[b,32],[b,32] -> [b,64]
            attn = torch.cat([attn, h], dim=1)

            #把每一步的记忆输出成词
            #[b,64] -> [b,27]
            out = self.out_fc(attn)
            outs.append(out)

        #把所有的输出词组合成一句话
        outs = torch.stack(outs, dim=0)
        #[10,b,27] -> #[b,10,27]
        outs = outs.permute(1, 0, 2)

        return outs


model = Attention()

out = model(sample[0], sample[1])
print(out[0, :2], out.shape)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

loss = 0
model.train()
for epoch in range(100):
    for i, data in enumerate(dataloader):
        x, y = data

        optimizer.zero_grad()

        #计算输出
        y_pred = model(x, y)

        #丢弃y的第一个词
        #因为训练的时候是以y的每一个词输入,预测下一个词
        #所以在计算loss的时候不需要第一个词
        #[b,11] -> [b,10]
        y = y[:, 1:]

        #打平,不然计算不了loss
        #[b,10,27] -> [b*10,27]
        y_pred = y_pred.reshape(-1, 27)

        #[b,10] -> [b*10]
        y = y.reshape(-1)

        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())


#构造反转的字典
reverse_zidian = {}
for k, v in zidian.items():
    reverse_zidian[v] = k


#数字化的句子转字符串
def seq_to_str(seq):
    seq = seq.detach().numpy()
    return ''.join([reverse_zidian[idx] for idx in seq])


seq_to_str(sample[0][0]), seq_to_str(sample[1][0])


# 预测
def predict(x):
    model.eval()

    # x编码
    # [b,8] -> [b,8,16]
    x = model.encoder_embed(x)
    # 进入循环网络,得到记忆
    # [b,8,16] -> [b,8,32],[1,b,32],[1,b,32]
    out_x, (h, c) = model.encoder(x, None)

    # [1,b,32],[1,b,32] -> [b,32],[b,32]
    h = h.squeeze()
    c = c.squeeze()

    # 初始化输入,每一个词的输入应该是上一个词的输出
    # 因为我们的y第一个词固定是<SOS>,所以直接以这个词开始
    # [b]
    out = torch.full((x.size(0),), zidian['<SOS>'], dtype=torch.int64)
    # [b] -> [b,16]
    out = model.decoder_embed(out)

    # 循环生成9个词,收尾的两个标签没有预测的价值,直接忽略了
    outs = []
    for i in range(9):
        # [b,32] -> [b,1,32]
        attn = model.get_attn(out_x, h)

        # 把每个词输入循环网络
        # 第一个词的记忆是x的最后一个词的记忆
        # 往后每个词的记忆是上一个词的记忆
        # [b,16] -> [b,32],[b,32]
        h, c = model.decoder_cell(out, (h, c))

        # [b,32],[b,32] -> [b,64]
        attn = torch.cat([attn, h], dim=1)

        # 把每一步的记忆输出成词
        # [b,64] -> [b,27]
        out = model.out_fc(attn)

        # 把每一步的记忆输出成词
        # [b,27] -> [b]
        out = out.argmax(dim=1)
        outs.append(out)

        # 把这一步的输出作为下一步的输入
        # [b] -> [b,16]
        out = model.decoder_embed(out)

    # 把所有的输出词组合成一句话
    # [9,b]
    outs = torch.stack(outs, dim=0)
    # [9,b] -> [b,9]
    outs = outs.permute(1, 0)

    return outs


# 测试
for i, data in enumerate(dataloader):
    x, y = data
    y_pred = predict(x)
    for xi, yi, pi in zip(x, y, y_pred):
        print(seq_to_str(xi), seq_to_str(yi), seq_to_str(pi))
    break


































