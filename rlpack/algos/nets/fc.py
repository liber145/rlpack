import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, idim, odim, hidden_list, device):
        super(FC, self).__init__()
        if len(hidden_list):
            self.linears = nn.ModuleList([nn.Linear(idim, hidden_list[0])])
            for i in range(1,len(hidden_list)):
                self.linears.append(nn.Linear(hidden_list[i-1], hidden_list[i]))
            self.out = nn.Linear(hidden_list[-1], odim).cuda()
        else:
            self.linears = nn.ModuleList([])
            self.out = nn.Linear(idim, odim).to(device)
        self.linears = self.linears.to(device)

    def forward(self, x):
        for fc in self.linears:
            x = fc(x)
            x = F.relu(x)
        x = self.out(x)
        return x