import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_forecasting(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, num_layers = 1, r_model = 'LSTM'):
        super(RNN_forecasting, self).__init__()
        if r_model == 'LSTM':
            self.rnn = nn.LSTM(input_dim, input_dim, num_layers, batch_first=True)
        elif r_model == 'GRU':
            self.rnn = nn.GRU(input_dim, input_dim, num_layers, batch_first=True)
        else:
            print('No Model')
            exit()
        
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x, _ = self.rnn(x)  #out : (batch_size, sequence_len, hidden_size)
        out = self.fc(x.permute(0,2,1)).permute(0,2,1)
        
        return out