# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, states, actions, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.states = states
        self.actions = actions
        self.data_num = self.states.shape[0]
        self.indices = list(range(self.data_num))
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.indices)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx: self.idx+self.batch_size]
        data = self.states[index, :, :]
        target = self.actions[index, :, :]
        self.idx += self.batch_size
        return torch.from_numpy(data), torch.from_numpy(target)

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data, fake_data, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        self.real_data = real_data
        self.fake_data = fake_data
        self.data = np.concatenate((self.real_data, self.fake_data), axis=0).tolist()
        self.labels = [1 for _ in range(self.real_data.shape[0])] +\
                        [0 for _ in range(self.fake_data.shape[0])]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx:self.idx+self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.DoubleTensor(np.asarray(data))
        label = torch.DoubleTensor(np.asarray(label))
        self.idx += self.batch_size
        return data, label


