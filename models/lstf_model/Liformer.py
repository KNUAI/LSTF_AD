import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Liformer(nn.Module):
    def __init__(self, seq_len, pred_len, input_size):
        super(Liformer, self).__init__()
        self.pred_len = pred_len
        self.attn = nn.MultiheadAttention(input_size, 1, batch_first=True)

    def forward(self, x):
        x_out = x
        dec_in = x[:, -1:, :]
        #y_out = self.dec(dec_in.permute(0,2,1)).permute(0,2,1)
        
        dec_inp = torch.zeros([x.shape[0], self.pred_len-1, x.shape[-1]]).float().cuda()
        dec_inp = torch.cat([dec_in, dec_inp], dim=1).float().cuda()
        
        out, _ = self.attn(dec_inp, x_out, x_out)
        #out = self.fc(out)

        return out
'''


class Liformer(nn.Module):
    def __init__(self, seq_len, pred_len, input_size):
        super(Liformer, self).__init__()
        self.avg = nn.AvgPool1d(2,2)
        self.conv = nn.Conv1d(seq_len//2, pred_len, kernel_size=1)
        self.LN = nn.LayerNorm([pred_len, input_size])

    def forward(self, x):
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.avg(x.permute(0,2,1)).permute(0,2,1)
        x = self.conv(x)
        x = self.LN(x)
        x = x + seq_last

        return x



