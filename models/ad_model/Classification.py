import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, input_size, latent_size, num_class, max_len):
        super(DNN, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, latent_size*4),
            nn.LeakyReLU(0.9)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(latent_size*4, latent_size*2),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2, latent_size),
            nn.LeakyReLU(0.9)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, num_class),
            nn.LeakyReLU(0.9)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear1(x.view(x.shape[0], -1))
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.classifier(x)

        return self.logsoftmax(x)

'''
class CNN(nn.Module):
    def __init__(self, input_size, latent_size, num_class, max_len):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, latent_size, 3, 2, 1),
            nn.LeakyReLU(0.9)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(latent_size, latent_size*2, 3, 2, 1),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2 * (input_size//4), latent_size),
            nn.LeakyReLU(0.9)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, num_class),
            nn.LeakyReLU(0.9)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x.view(x.shape[0], 1, -1))
        x = self.conv2(x)
        x = self.linear3(x.view(x.shape[0], -1))
        x = self.classifier(x)

        return self.logsoftmax(x)
'''

class CNN(nn.Module):
    def __init__(self, input_size, latent_size, num_class, max_len):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, latent_size, 3, 1, 1),
            nn.LeakyReLU(0.9)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(latent_size, latent_size*2, 3, 1, 1),
            nn.LeakyReLU(0.9)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(latent_size*2 * (input_size//4), latent_size),
            nn.LeakyReLU(0.9)
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, num_class),
            nn.LeakyReLU(0.9)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.max_pool1d(self.conv1(x.view(x.shape[0], 1, -1)), 2)
        x = F.max_pool1d(self.conv2(x), 2)
        x = self.linear3(x.view(x.shape[0], -1))
        x = self.classifier(x)

        return self.logsoftmax(x)


