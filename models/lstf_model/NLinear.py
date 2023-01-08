import torch
import torch.nn as nn
import torch.nn.functional as F

# NLinear_model
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]


