import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

class FC(nn.Module):
    def __init__(self, idim, odim, hidden_list, device, activation=None):
        super(FC, self).__init__()
        if len(hidden_list):
            self.linears = nn.ModuleList([nn.Linear(idim, hidden_list[0])])
            for i in range(1,len(hidden_list)):
                self.linears.append(nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.out = nn.Linear(hidden_list[-1], odim).to(device)
        else:
            self.linears = nn.ModuleList([])
            self.out = nn.Linear(idim, odim).to(device)
        self.linears = self.linears.to(device)
        self.output_act = activation

        self.apply(weights_init_)

    def forward(self, x):
        for fc in self.linears:
            x = fc(x)
            x = F.relu(x)
        x = self.out(x)
        if self.output_act is not None:
            x = self.output_act(x)
        return x

class Ensemble_FC(nn.Module):
    def __init__(self, sdim, adim, odim, hidden_list, device):
        super(Ensemble_FC, self).__init__()
        if len(hidden_list):
            s_embed = (hidden_list[0]*sdim)//(sdim+adim)
            a_embed = hidden_list[0]-s_embed
            self.linears = nn.ModuleList([nn.Linear(sdim, s_embed)])
            self.linears.append(nn.Linear(adim, a_embed))
            for i in range(1, len(hidden_list)):
                self.linears.append(nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.out = nn.Linear(hidden_list[-1], odim).to(device)
        else:
            self.linears = nn.ModuleList([])
            self.out = nn.Linear(sdim+adim, odim).to(device)
        self.linears = self.linears.to(device)

    def forward(self, s, a):
        s_embed = self.linears[0](s)
        a_embed = self.linears[1](a)
        x = torch.cat((s_embed, a_embed), dim=-1)
        x = F.relu(x)
        for i in range(2, len(self.linears)):
            x = self.linears[i](x)
            x = F.relu(x)
        x = self.out(x)
        return x