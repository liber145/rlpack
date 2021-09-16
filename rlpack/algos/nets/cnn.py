import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

class CNN(nn.Module):
    def __init__(self, dim_obs, dim_act, hidden_list, device):
        super(CNN, self).__init__()

        self.cnn_layers = nn.ModuleList([
            nn.Conv2d(dim_obs[2], hidden_list[0][0], kernel_size=hidden_list[0][1], stride=hidden_list[0][2])
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_list[0][0])
        ])
        for i in range(1, len(hidden_list)):
            self.cnn_layers.append(
                nn.Conv2d(hidden_list[i-1][0], hidden_list[i][0], kernel_size=hidden_list[i][1], stride=hidden_list[i][2]))
            self.bn_layers.append(nn.BatchNorm2d(hidden_list[i][0]))
        self.bn_layers.to(device)
        self.cnn_layers.to(device)
        self.fc1 = nn.Linear(7*7*64, 512).to(device)
        self.fc2 = nn.Linear(512, dim_act).to(device)

    def forward(self, x):
        x = x.float()/255
        for cnn, bn in zip(self.cnn_layers, self.bn_layers):
            x = cnn(x)
            x = F.relu(bn(x))
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = self.fc2(x)
        return x