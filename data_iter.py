# -*- coding:utf-8 -*-

import os
import random
import math

import tqdm

import numpy as np
import torch
class GenDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, data, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data = data
        self.data_num = self.data.shape[0]
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0


    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        d = self.data[self.idx: self.idx+self.batch_size, :, ]
        d = torch.DoubleTensor(d)
        data = torch.cat([torch.zeros(self.batch_size, d.shape[1], d.shape[2]).double(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, d.shape[1], d.shape[2]).double()], dim=1)
        self.idx += self.batch_size
        return data, target

class DisDataIter(object):
    """ Toy data iter to load digits"""
    def __init__(self, real_data, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        # real_data_lis = self.read_file(real_data_file)
        self.real_data = real_data
        fake_data_lis = self.read_file(fake_data_file)
        self.data = self.real_data + fake_data_lis
        self.labels = [1 for _ in range(self.real_data.shape[0])] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = zip(self.data, self.labels)
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0

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
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label


