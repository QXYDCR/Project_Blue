from Util import *
from model import *


# ------------------------ 5、定义网络模型 ------------------------

class Actor(nn.Module):
    def __init__(self,embedding_dim=50, hidden_size=50,n_class=2, batch = 64):
        super(Actor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.batch = batch

        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, x, h, c):
        # h, c: [1, 100], x: [b, 50]
        h, c = self.lstm_cell(x, (h, c))
        out = self.linear(c)
        return h, c, out

    def init_H_C(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(self.batch, self.hidden_size)
        c = torch.zeros(self.batch, self.hidden_size)
        return h, c

actor = Actor()
opti_actor = torch.optim.Adam(actor.parameters(), lr = 0.001)


class Critic(nn.Module):
    def __init__(self,embedding_dim=50, hidden_size=50,n_class=2, batch = 64):
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.batch = batch

        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, n_class)

    def forward(self, x, h, c):
        # x: [1, ] --> [1, 50]
        x = self.embedding(x)
        # h, c: [1, 100]
        h, c = self.lstm_cell(x, (h, c))
        out = self.linear(c)
        return h, c, out

    def init_H_C(self):
        # h, c [batch_size, n_hidden]
        h = torch.zeros(self.batch, self.hidden_size)
        c = torch.zeros(self.batch, self.hidden_size)
        return h, c

critic = Critic()
loss_func_critic = nn.CrossEntropyLoss()
opti_critic = torch.optim.Adam(critic.parameters(), lr = 0.001)

class FC_Classifier(nn.Module):
    def __init__(self, hidden_size = 50, n_class = 2):
        super(FC_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_class = n_class

        self.fc = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        x = self.fc(x)
        return x

FC = FC_Classifier()
loss_FC_func = nn.CrossEntropyLoss()
optimizer_FC = torch.optim.Adam(FC.parameters(), lr=0.001)


class WS_LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embed_size = 50, hidden_size = 50, n_class = 2,
                 finetune_emb=False, layer_num = 1, batch = 64, epochs=5, learning_rate=0.001, l2reg=1e-4,
                 dropout=0.1, max_len = 40):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(WS_LSTM_Model, self).__init__()
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


    def forward(self, x, actor, critic):
        # x: [b, input_size], h, c: [b, hidden_size]
        # x: [b, input_size] --> [b, time_steps, embed_size] --> [time_steps, b, embed_size]
        # 用于记录句子的成分, 没有被向量化的原始句子序号
        lines = x
        x = self.embedding(x)
        x = x.transpose(1, 0)

        h_a, c_a = self.init_h_c()
        h_c, c_c = self.init_h_c()
        actions_pair = []
        pred_actions = []
        for j in range(x.shape[0]):
            # [batch, input_size]
            # h_c作为状态，一直保存且跟新
            h_a = h_c
            # 选择动作
            h_a, c_a, out = actor(x[j], h_a, c_a)

            out = torch.log_softmax(out, 1)
            actions_pair.append(out)
            pred_actions.append(torch.argmax(out, dim=1))



        return 11


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


        actor = Actor()
        critic = Critic()
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

