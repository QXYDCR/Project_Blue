import torch

from Util import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from senti_demo import *



class WS_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, id2word, embed_size = 50, hidden_size = 50, n_class = 2,
                 finetune_emb=False, layer_num = 1, batch = 64, epochs=5, learning_rate=0.001, l2reg=1e-4,
                 dropout=0.3, max_len = 40):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(WS_LSTM, self).__init__()
        self.embed_size = embed_size        # 嵌入维度
        self.hidden_size = hidden_size      # 隐藏层维度
        self.layer_num = layer_num          # LSTM层数
        self.batch = batch                  # 批量数
        self.max_len = max_len
        self.id2word = id2word

        self.mc = MySentiWorld(id2word)

        # 加载词向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = finetune_emb

        # 网络构成
        self.lstmcell = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)
        self.dropout = nn.Dropout(p = 0.1)
        # 其他属性
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2reg = l2reg
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):
        ff = x      # 记录数值化的文本
        # x: [b, input_size], h, c: [b, hidden_size]
        # x: [b, input_size] --> [b, time_steps, embed_size] --> [time_steps, b, embed_size]
        x = self.embedding(x)
        x = x.transpose(1, 0)
        h, c = self.init_h_c()

        out_final = 0
        outputs = torch.zeros((40, 64, 50))
        actions_pair = torch.zeros((40, 64, 2)) # 预测动作
        pred_actions = torch.zeros((40, 64))    # 回合预测动作对
        true_actions = torch.zeros((40, 64))    # 真实动作

        for i in range(x.shape[0]):
            # [batch, input_size]
            h, c = self.lstmcell(x[i, :, :], (h, c))

            out_final = c
            outputs[i] = c

            out = self.linear(c)
            # [b, 2]
            out = torch.log_softmax(out, 1)
            # 取得神经网络输出的动作概率对 (0,3, 0.7)
            # actions_pair.append(out)
            actions_pair[i] = out
            # 取得神经网络输出的动作对 (0， 1)
            # pred_actions.append(torch.argmax(out, dim=1))
            pred_actions[i] = torch.argmax(out, dim=1).type(torch.LongTensor)

            # 取得批量词语, [b, ]
            w_s = get_words_from_num(ff[:, i], self.id2word)

            # 取得真实动作, [b, ]
            temp_true_actions = get_sent_by_words(w_s)
            true_actions[i] = torch.LongTensor(temp_true_actions)

        # 跟新参数
        # [b, 40] ndarray --> tensor
        episode_score = self.mc.get_batch_scores(ff)
        episode_score = torch.FloatTensor(episode_score)


        # 计算回合的得分, 每个步骤都有分
        # I: score, pred_actions: [40, 64], scores:[b, 40]
        batch_episode_scores = get_batch_episode_scores(episode_score, pred_actions)
        batch_episode_scores = torch.FloatTensor(batch_episode_scores)

        # actions_pair [40, 64, 2] --> [64, 40, 2] --> [64 * 40, 2]
        t_actions_pair = actions_pair.permute(1, 0, 2).reshape(-1, 2)
        # [40, b] --> [b, 40]
        t_pred_actions = pred_actions.permute(1, 0).reshape(-1).type(torch.LongTensor)

        # print(t_pred_actions.shape, type(t_pred_actions))
        selected_logprobs = self.criterion(t_actions_pair, t_pred_actions)

        loss_actor = -(batch_episode_scores.reshape(-1, 40) * selected_logprobs).mean()
        # print(scores)
        # 更新FC
        # [40, b, 50] --> [b, 40, 50]
        outputs = outputs.permute(1, 0, 2)
        # out: [size]: [b, 50] * [50, 2] = [b, 2]
        out_final = self.linear(out_final)
        # out_final = self.dropout(out_final)

        return loss_actor, out_final


    def init_h_c(self):
        # h, c [batch_size, n_hidden]
        h = torch.randn((self.batch, self.hidden_size))
        c = torch.randn((self.batch, self.hidden_size))
        return h, c

    def compute_loss(self, x, y):
        '''
            Computing loss and evaluation metrics for predictions.
            inputs:
                - x (torch.tensor):      input tensor for neural network
                - y (torch.tensor):      label tensor
            return:
                - loss (torch.float):    binary cross-entropy loss (CE) between LSTM(x) and y
                - accuracy (float):      accuracy of predictions (sklearn)
        '''

        loss_actor, predictions = self.forward(x)
        # y, [64, 2] --> [64,]
        loss = self.criterion(predictions, y.squeeze())
        outputs = torch.argmax(predictions, 1)
        accuracy = accuracy_score(y.squeeze(), outputs)

        loss = loss
        return loss, accuracy


    def evaluate_loader(self, loader):
        '''
            Computing loss and evaluation metrics for a specific torch.loader.
            inputs:
                - loader (torch.loader):    dataset in torch.loader format
            return:
                - metrics (dict):           mapping of metric name (str) to metric value (float)
        '''
        # compute loss and accuracy for that loader
        metrics = {'loss': 0, 'accuracy': 0, }
        # loop over examples of loader
        for i, (x, y) in enumerate(loader):
            loss, accuracy = self.compute_loss(x, y)
            # sum up metrics in dict
            metrics['loss'] += loss.item()
            metrics['accuracy'] += accuracy
        # normalize all values
        for k in metrics.keys():
            metrics[k] /= len(loader)

        return metrics

    def train_data(self, train_loader, val_loader, freq_prints=5):
        '''
        Fit a classifier with train and val loaders.
        inputs:
            - train_loader (torch.loader):     training set in torch.loader format
            - val_loader (torch.loader):       validation set in torch.loader format
            - freq_prints (int):               frequency of printing performances of training
        return:
            - history (dict):                  metrics values (metric name to values)
        '''
        history = {'loss': [], 'val_loss': [],'accuracy': [], 'val_accuracy': [],}

        for epoch in range(self.epochs):
            # one epoch
            train_metrics = {'loss': 0, 'accuracy': 0, }
            for i, (x, y) in enumerate(train_loader):

                # Forward + Backward + Optimize

                loss, accuracy = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # 归一化分数
            for k in train_metrics.keys():
                train_metrics[k] /= len(train_loader)

            # 计算验证集的准确性
            val_metrics = self.evaluate_loader(val_loader)

            # save metrics in history
            for key in train_metrics:
                history[key].append(train_metrics[key])
            for key in val_metrics:
                history['val_' + key].append(val_metrics[key])

                # printing of performance at freq_prints frequency
                if (epoch + 1) % 1 == 0:
                    print("Epoch {}/{}\nTrain performance: loss={:.3f}, accuracy={:.3f}".format(
                            epoch + 1, self.epochs, history['loss'][-1], history['accuracy'][-1]))

        return history



