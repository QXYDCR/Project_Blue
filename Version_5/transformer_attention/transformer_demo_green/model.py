import data
import torch
import torch.nn as nn
import numpy as np
import Utils
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embed = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = nn.Linear(32, 27)

        self.loss_func = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x, y):
        # [b, 1, 11, 11]
        mask_x = Utils.mask_x(x)
        mask_y = Utils.mask_y(y)

        # 编码,添加位置信息
        # x = [b, 11] -> [b, 11, 32]
        # y = [b, 11] -> [b, 11, 32]
        x, y = self.embed(x), self.embed(y)

        # 编码层计算
        # [b, 11, 32] -> [b, 11, 32]
        x = self.encoder(x, mask_x)

        # 解码层计算
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = self.decoder(x, y, mask_x, mask_y)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.fc_out(y)

        return y

class PositionEmbedding(nn.Module):
    """
    用于将数据进行嵌入和加入位置参数
    """
    def __init__(self, max_len=11, emb_dim=32, n_vocab=27):
        """
        :param max_len: (int), 数据的最大长度
        :param emb_dim: (int), 嵌入的维度
        :param n_vocab: (int), 词表的大小
        """
        super().__init__()
        # 完成对位置参数的编码
        # 数据为[max_len, emb_dim], (pos)对每个位置[0],[1],...[10]的32个嵌入维度插入位置参数
        pos = np.expand_dims(np.arange(max_len), 1)  # [max_len, 1]

        # pos [max_len, emb_dim], [np.arange(emb_dim) // 2] = [0, 0, 1, 1,...]
        pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(emb_dim) // 2, 0) / emb_dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)

        # 嵌入层
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        # 向量初始化
        self.embeddings.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # CPU ? GPU
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        # x: [b, 11] --> [b, 11, 32]
        x_embed = self.embeddings(x.long())
        x_embed = x_embed + self.pe  # [n, step, emb_dim]
        return x_embed  # [batch, seq_len, emb_dim],


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = nn.Linear(32, 32)
        self.fc_K = nn.Linear(32, 32)
        self.fc_V = nn.Linear(32, 32)
        #
        self.out_fc = nn.Linear(32, 32)
        # 层正则化
        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, Q, K, V, mask):
        # b句话,每句话11个词,每个词编码成32维向量
        # Q,K,V = [b, 11, 32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        original_Q = Q

        # 线性运算,维度不变, 参数矩阵K, V, Q, [32, 32]
        # [b, 11, 32] -> [b, 11, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 11, 32] -> [b, 4, 11, 8]
        Q = Q.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 11, 4, 8).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 11, 8] -> [b, 11, 32]
        score = Utils.attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        score = F.dropout(self.out_fc(score), 0.1)

        # 短接,规范化, 残差+层正则化
        score = self.norm(original_Q + score)
        return score


# 全连接输出层
class FullyConnectedOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(p=0.1)
        )
        # 层正则化
        self.norm = nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

    def forward(self, x):
        # 线性全连接运算
        # [b, 11, 32] * [32, 128] * [128, 32] -> [b, 11, 32]
        out = self.fc(x)

        # add + Norm, 残差连接 + 层正则化
        out = self.norm(x + out)

        return out


class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,属于自注意力层, 维度不变
        # [b, 11, 32] -> [b, 11, 32]
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        out = self.fc(score)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_x, mask_y):
        # 先计算y的自注意力,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.mh1(y, y, y, mask_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = self.mh2(y, x, x, mask_x)

        # 全连接输出,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        y = self.fc(y)

        return y


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_x, mask_y):
        y = self.layer_1(x, y, mask_x, mask_y)
        y = self.layer_2(x, y, mask_x, mask_y)
        y = self.layer_3(x, y, mask_x, mask_y)
        return y



model = Transformer()

def predict(x):
    # x = [b, 11]
    model.eval()

    # [b, 1, 11, 11]
    mask_x = Utils.mask_x(x)

    # 初始化输出,这个是固定值
    # [b, 12]
    # [[25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    target = [data.zidian["<SOS>"]] + [data.zidian["<PAD>"]] * 11
    target = torch.LongTensor(target).unsqueeze(0)

    # x编码,添加位置信息
    # [b, 11] -> [b, 11, 32]
    x = model.embed(x)

    # 编码层计算,维度不变
    # [b, 11, 32] -> [b, 11, 32]
    x = model.encoder(x, mask_x)

    # 遍历生成第0个词到第11个词
    for i in range(11):
        # 丢弃target中的最后一个词
        # 因为计算时,是以当前词,预测下一个词,所以最后一个词没有用
        # [b, 11]
        y = target[:, :-1]

        # [b, 1, 11, 11]
        mask_y = Utils.mask_y(y)

        # y编码,添加位置信息
        # [b, 11] -> [b, 11, 32]
        y = model.embed(y)

        # 解码层计算,维度不变
        # [b, 11, 32],[b, 11, 32] -> [b, 11, 32]
        y = model.decoder(x, y, mask_x, mask_y)

        # 全连接输出,27分类
        # [b, 11, 32] -> [b, 11, 27]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [b, 11, 27] -> [b, 27]
        out = out[:, i, :]

        # 取出分类结果
        # [b, 27] -> [b]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    model.train()
    return target


def train():
    for i in range(100):
        for batch_i, (x, y) in enumerate(data.get_dataloader()):
            # x = [b, 11]
            # x = 05-06-15<PAD><PAD><PAD>
            # y = [b, 12]
            # y = <SOS>15/Jun/2005<EOS><PAD>

            model.optim.zero_grad()

            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字典
            pred = model(x, y[:, :-1])

            loss = model.loss_func(pred.reshape(-1, 27), y[:, 1:].reshape(-1))
            loss.backward()
            model.optim.step()

            if batch_i % 50 == 0:
                pred = data.seq_to_str(predict(x[0:1])[0])
                print(i, data.seq_to_str(x[0]), data.seq_to_str(y[0]), pred)


if __name__ == '__main__':
    # 测试PositionEmbedding
    # 83-01-17<PAD><PAD><PAD>
    # <SOS>17/Jan/1983<EOS><PAD>
    # b = data.get_sample()[0][:1, :]
    # cc = PositionEmbedding()
    # print(b.shape,cc(b))

    train()




















