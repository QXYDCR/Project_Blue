from Util import *
import torch.nn.functional as F
import math


class PositionEmbedding(nn.Module):
    def __init__(self, max_len = 40, emb_dim = 50, n_vocab = 27, embedding_matrix = None):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len), 1)  # [max_len, 1]

        # [max_len, emb_dim]
        pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(emb_dim) // 2, 0) / emb_dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        self.embeddings.weight.data.normal_(0, 0.1)

        # 加载词向量
        self.embedding = nn.Embedding(n_vocab, emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        # x: [b, 40] --> [b, 40, 50]
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)

        x_embed = self.embedding(x)

        x_embed = x_embed + self.pe  # [n, step, emb_dim]
        return x_embed  # [batch, seq_len, emb_dim],


# data = DateDataset()
# print(PositionEmbedding()(get_sample()[0]).shape)

def attention(Q, K, V, mask):
    # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 11, 8]
    #           [b, 5, 40, 10]
    # [b, 5, 40, 10] * [b, 5, 40, 10] -> [b, 5, 40, 40]
    # Q,K矩阵相乘,结果除以根号头数,这里完全是公式
    score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(10)

    # mask遮盖,mask是true的地方都被替换成-inf
    # mask = [b, 1, 11, 11]
    # score = score.masked_fill_(mask, -np.inf)
    score = F.softmax(score, dim=-1)

    # 这一步也是公式
    # [b, 5, 40, 40] * [b, 5, 40, 10] -> [b, 5, 40, 10]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 5, 40, 10] -> [b, 40, 50]
    score = score.permute(0, 2, 1, 3).reshape(-1, 40, 50)

    return score


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = nn.Linear(50, 50)
        self.fc_K = nn.Linear(50, 50)
        self.fc_V = nn.Linear(50, 50)

        self.out_fc = nn.Linear(50, 50)

        # 规范化之后,均值是0,标准差是1,前提是没有经过线性运算的话
        # mean = out.mean(dim=(0, 2))
        # std = out.std(dim=(0, 2))
        # BN是取不同样本的同一个通道的特征做归一化
        # LN取的是同一个样本的不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # self.norm = nn.BatchNorm1d(num_features=11, affine=True)
        self.norm = nn.LayerNorm(normalized_shape=50, elementwise_affine=True)

    def forward(self, Q, K, V, mask):
        # b句话,每句话11个词,每个词编码成32维向量
        # Q,K,V = [b, 11, 32]
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        original_Q = Q

        # 线性运算,维度不变
        # [b, 11, 32] -> [b, 11, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话11个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 40, 50] -> [b, 5, 40, 10]
        Q = Q.reshape(b, 40, 5, 10).permute(0, 2, 1, 3)
        K = K.reshape(b, 40, 5, 10).permute(0, 2, 1, 3)
        V = V.reshape(b, 40, 5, 10).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 5, 40, 10] -> [b, 40, 50]
        score = attention(Q, K, V, mask)

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
            nn.Linear(in_features=50, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=50),
            nn.Dropout(p=0.1)
        )

        self.norm = nn.LayerNorm(normalized_shape=50, elementwise_affine=True)

    def forward(self, x):
        # 线性全连接运算
        # [b, 40, 50] -> [b, 40, 50]
        out = self.fc(x)

        # 做短接,正规化
        out = self.norm(x + out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
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


class Transformer(nn.Module):
    def __init__(self, n_vocab = None, embed_matrix = None):
        super().__init__()
        self.embed = PositionEmbedding(n_vocab = n_vocab, embedding_matrix = embed_matrix)
        self.encoder = Encoder()
        self.liner = nn.Linear(50, 50)
        self.fc_out = nn.Linear(50, 2)

        self.decoder = nn.LSTM(50, 50)

    def forward(self, x):

        # 编码,添加位置信息 == embed + position
        # x = [b, 40] -> [b, 40, 50]
        # y = [b, 40] -> [b, 40, 50]
        x = self.embed(x)

        # 编码层计算
        # [b, 11, 32] -> [b, 11, 32]
        x = self.encoder(x, 0)

        # 全连接输出,维度不变
        # [b, 40, 50] -> [b, 40, 50]
        y = self.liner(x)

        y = y.permute(1, 0, 2)
        outputs, (h, c) = self.decoder(y)

        # [40, 64, 50] --> [64, 50]
        outputs = outputs[-1]  # [batch_size, n_hidden]
        # [64, 50] --> [64, 2]
        outputs = self.fc_out(outputs)

        return outputs















