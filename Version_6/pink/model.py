from Util import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embed_size = 50, hidden_size = 50, n_class = 2,
                 finetune_emb=False, layer_num = 1, batch = 64, epochs=5, learning_rate=0.001, l2reg=1e-4,
                 dropout=0.1, max_len = 40):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(LSTM_Model, self).__init__()
        self.embed_size = embed_size        # 嵌入维度
        self.hidden_size = hidden_size      # 隐藏层维度
        self.layer_num = layer_num          # LSTM层数
        self.batch = batch                  # 批量数
        self.max_len = max_len
        # 加载词向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = finetune_emb

        # 网络构成
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

        # 其他属性
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2reg = l2reg
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)


    def forward(self, x, h, c):
        # x_size,[batch, input_size],[128, 40] --> [batch, input_size, embed_size]
        # [128, 40] --> [64, 40, 50] --> [40, 64, 50]
        x = self.embedding(x)
        x = x.transpose(0, 1)

        outputs, (h, c) = self.lstm(x, (h, c))
        # [40, 64, 50] --> [64, 50]
        outputs = outputs[-1]  # [batch_size, n_hidden]
        # [64, 50] --> [64, 2]
        outputs = self.linear(outputs)

        return outputs

    def init_h_c(self):
        # D * \text{num\_layers}, N, H_{out}
        h = torch.zeros(self.layer_num, self.batch, self.hidden_size)
        c = torch.zeros(self.layer_num, self.batch, self.hidden_size)
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
        h, c = self.init_h_c()
        # [b, 40], [1, b, hidden_size] --> [b, n_class]
        predictions = self.forward(x, h, c)
        # y, [64, 1] --> [64,]
        loss = self.criterion(predictions, y.squeeze())
        outputs = torch.argmax(predictions, 1)
        accuracy = accuracy_score(y.squeeze(), outputs)

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
                self.optimizer.zero_grad()  # zero the gradient buffer

                loss, accuracy = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy

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


class LSTM_Cell_Model(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embed_size = 50, hidden_size = 50, n_class = 2,
                 finetune_emb=False, layer_num = 1, batch = 64, epochs=5, learning_rate=0.001, l2reg=1e-4,
                 dropout=0.1, max_len = 40):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(LSTM_Cell_Model, self).__init__()
        self.embed_size = embed_size        # 嵌入维度
        self.hidden_size = hidden_size      # 隐藏层维度
        self.layer_num = layer_num          # LSTM层数
        self.batch = batch                  # 批量数
        self.max_len = max_len
        # 加载词向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = finetune_emb
        # 网络构成
        self.lstmcell = nn.LSTMCell(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)
        # 其他属性
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2reg = l2reg
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def forward(self, x):
        # x: [b, input_size], h, c: [b, hidden_size]
        # x: [b, input_size] --> [b, time_steps, embed_size] --> [time_steps, b, embed_size]
        x = self.embedding(x)
        x = x.transpose(1, 0)
        h, c = self.init_h_c()

        for j in range(x.shape[0]):
            # [batch, input_size]
            h, c = self.lstmcell(x[j], (h, c))
        out = self.linear(c)

        return h, c, out


    def init_h_c(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(self.batch, self.hidden_size)
        c = torch.zeros(self.batch, self.hidden_size)
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

        _, _, predictions = self.forward(x)
        # y, [64, 1] --> [64,]
        loss = self.criterion(predictions, y.squeeze())
        outputs = torch.argmax(predictions, 1)
        accuracy = accuracy_score(y.squeeze(), outputs)

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
                self.optimizer.zero_grad()  # zero the gradient buffer

                loss, accuracy = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy

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


class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embed_size = 50, hidden_size = 50, n_class = 2,
                 finetune_emb=False, layer_num = 1, batch = 64, epochs=5, learning_rate=0.001, l2reg=1e-4,
                 dropout=0.1, max_len = 40):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(LSTM_Attention, self).__init__()
        self.embed_size = embed_size        # 嵌入维度
        self.hidden_size = hidden_size      # 隐藏层维度
        self.layer_num = layer_num          # LSTM层数
        self.batch = batch                  # 批量数
        self.max_len = max_len
        # 加载词向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = finetune_emb

        # 网络构成
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size = 50,
                               hidden_size=hidden_size,
                               num_layers=2,
                               batch_first=False,
                               bidirectional=True,
                               dropout = 0.3)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.decoder = nn.Linear(2 * hidden_size, 2)

        # 其他属性
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2reg = l2reg
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)


    def forward(self, x, h, c):
        # x_size,[batch, input_size],[128, 40] --> [batch, input_size, embed_size]
        # [128, 40] --> [64, 40, 50] --> [40, 64, 50]
        x = self.embedding(x)
        x = x.transpose(0, 1)

        # [40, 64, 50 * 2]
        outputs, _ = self.encoder(x)
        # outputs形状是[s_l, b, 2 * num_hiddens) --> (b, s_l, 2 * num_hiddens]
        x = outputs.permute(1, 0, 2)

        # Attention过程
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        u = torch.tanh(torch.matmul(x, self.w_omega))

        # att形状是(batch_size, seq_len, 1)
        att = torch.matmul(u, self.u_omega)

        # att_score形状仍为(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)

        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        scored_x = x * att_score

        feat = torch.sum(scored_x, dim=1)
        # feat形状是(batch_size, 2 * num_hiddens)
        outs = self.decoder(feat)
        # out形状是(batch_size, 2)

        return outs

    def init_h_c(self):
        # D * \text{num\_layers}, N, H_{out}
        h = torch.zeros(self.layer_num, self.batch, self.hidden_size)
        c = torch.zeros(self.layer_num, self.batch, self.hidden_size)
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
        h, c = self.init_h_c()
        # [b, 40], [1, b, hidden_size] --> [b, n_class]
        predictions = self.forward(x, h, c)
        # y, [64, 1] --> [64,]
        loss = self.criterion(predictions, y.squeeze())
        outputs = torch.argmax(predictions, 1)
        accuracy = accuracy_score(y.squeeze(), outputs)

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
                self.optimizer.zero_grad()  # zero the gradient buffer

                loss, accuracy = self.compute_loss(x, y)
                train_metrics['loss'] += loss.item()
                train_metrics['accuracy'] += accuracy

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




























































































































































































