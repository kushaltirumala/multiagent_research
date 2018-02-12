import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, gpu=True, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense = nn.Linear(hidden_dim + action_dim, 1)
            
    def forward(self, x, a, h=None):  # x: seq * batch * 22, a: seq * batch * 22
        p, hidden = self.gru(x, h)   # p: seq * batch * 22
        p = torch.cat([p, a], 2)   # p: seq * batch * 44
        prob = F.sigmoid(self.dense(p))    # prob: seq * batch * 1
        return prob, hidden

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))