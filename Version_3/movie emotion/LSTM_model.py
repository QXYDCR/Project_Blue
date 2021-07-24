from Utils import *


# class LSTM_Model(nn.Module):
#     def __int__(self, vocab_size, embedding_dim = 50, hidden_size = 50,
#                 pretrained_embed, is_updata_w2c, n_class):
#         super(LSTM_Model, self).__int__()
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.hidden_size = hidden_size
#         self.pretrained_embed = pretrained_embed
#         self.is_updata_w2c = is_updata_w2c
#         self.n_class = n_class
#
#         # 使用预训练的词向量
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
#         self.embedding.weight.requires_grad = is_updata_w2c
#
#         self.lstm = nn.LSTM(embedding_dim, hidden_size)
#         self.linear = nn.Linear(hidden_size, n_class)
#
#     def forward(self, x, h, c):
#         # x: [32, 50] --> [32, 50, 50]
#         x = self.embedding(x)
#         x = x.transpose(0, 1)
#
#         outputs, h, c = self.lstm(x, (h, c))
#         # [32, 50, 50] --> [32, 50]
#         outputs = outputs[-1]  # [batch_size, n_hidden]
#         # [32, 50] --> [32, 2]
#         outputs = self.linear(outputs)
#         return outputs, h, c
#
#     def init_H_C(self):
#         h = torch.zeros(1, 32, self.hidden_size)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#         c = torch.zeros(1, 32, self.hidden_size)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#         return h, c
















