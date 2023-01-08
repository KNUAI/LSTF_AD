import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN_model
class RNN_ad(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, max_len, r_model):
        super(RNN_ad, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        if r_model == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, n_layer, bidirectional=True, batch_first=True)
        elif r_model == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, n_layer, bidirectional=True, batch_first=True)
        else:
            print('No Model')
            exit()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, input_size),
            nn.LeakyReLU(0.9)
        )

    def forward(self, x):
        outs, _ = self.rnn(x)
        out = self.fc(outs[:, -1, :])

        return out