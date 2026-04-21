import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim=100,
                 num_classes=2,
                 hidden_size=32,
                 lstm_layer_num=2,
                 attention_dim=64,
                 dropout_rate=0.2,
                 fc_dim=100):

        super(BiLSTMAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.LSTM_layer_num = lstm_layer_num
        self.attention_dim = attention_dim
        self.dropout = dropout_rate
        self.fc_dim = fc_dim

        # Bi-LSTM
        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.LSTM_layer_num,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

        # Fully connected layer
        self.d1 = nn.Linear(self.hidden_size * 2, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)

        # Attention Layer
        self.attention_fc = nn.Linear(self.hidden_size * 2, self.attention_dim)
        self.attention_out = nn.Linear(self.attention_dim, 1)

    def attention(self, lstm_output):
        attention_weights = torch.tanh(self.attention_fc(lstm_output))
        attention_weights = self.attention_out(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attention_output

    def forward(self, data):
        x = data[0]
        lstm_out, _ = self.bilstm(x)
        x = self.attention(lstm_out)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        out = F.softmax(x, dim=1)
        return out