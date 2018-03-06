# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, num_classes, state_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len * state_dim)
        """
        output, hidden = self.gru(x) 
        prob = self.softmax(self.lin(output))[:, 0, :]
        return prob

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
