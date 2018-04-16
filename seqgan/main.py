# -*- coding:utf-8 -*-

import os
import random
import math

import argparse
import tqdm

import numpy as np
import struct

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import Target_LSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from utils.draw_tools import plot_sequences
import pickle
import visdom
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--file', action='store', default=None)
opt = parser.parse_args()
print(opt)

draw_pretrained_discriminator_images= False

# -----------------------------------------
graph_pretrain_discriminator = None
graph_pretrain_generator = None
graph_adversarial_training = None
graph_adversarial_training_discriminator = None
graph_pretrain_generator_validation = None
graph_pretrain_discriminator_validation = None
experiment_num = 11

same_start_set = True


# Basic Training Paramters
SEED = 88
BATCH_SIZE = 32 
TOTAL_BATCH = 100
GENERATED_NUM = 96
VOCAB_SIZE = 22
PRE_EPOCH_NUM = 20
VAL_FREQ = 3

'''
if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True
'''

# Genrator Parameters
g_state_dim = 22
g_hidden_dim = 256
g_action_dim = 22
g_sequence_len = 69

# Discriminator Parameters
d_num_class = 2
d_state_dim = 22
d_hidden_dim = 256

vis = visdom.Visdom()
if not os.path.exists("saved_images/experiment_"+str(experiment_num)):
    os.makedirs("saved_images/experiment_"+str(experiment_num))


def load_model(path):
    print ("Loading learned model")
    generator, discriminator = pickle.load(open(path, 'rb'))
    return generator, discriminator

def save_model(generator, discriminator, path):
    print ("Saving Model")
    if opt.cuda:
        generator, discriminator = generator.cpu(), discriminator.cpu()
    pickle.dump((generator, discriminator), open(path, 'wb'))
    if opt.cuda:
        generator, discriminator = generator.cuda(), discriminator.cuda()

# draws FIRST trajectory for the OFFENSE team
def draw_samples(states, show_image=True, save_image=False, name=None):
    print ("Drawing")
    draw_data = states[0, :, 2:12]
    normal = [47.0, 50.0] * 5
    draw_data = draw_data * normal
    colormap = ['b', 'r', 'g', 'm', 'y']
    if show_image:
        plot_sequences([draw_data], macro_goals=None, colormap=colormap, show=show_image, burn_in=0)
    else:
        plot_sequences([draw_data], macro_goals=None, colormap=colormap, save_name="saved_images/experiment_"+str(experiment_num)+"/"+name+"_offense", show=False, burn_in=0)
        
def target_lstm_generate_samples(model, batch_size, generated_num):
    samples = []
    actions = []
    for _ in range(int(generated_num/batch_size)):
        temp = model.sample(batch_size, g_sequence_len)
        sample = temp[0].cpu().data.numpy()
        action = temp[1].cpu().data.numpy()
        samples.append(sample)
        actions.append(action)

    return np.vstack(samples), np.vstack(actions)      

def generate_samples(model, batch_size, generated_num, train_states, definite_start_state=None, return_start_states=False):
    samples = []
    exp_samples = []
    for _ in range(int(generated_num / batch_size)):
        # sample some sequences from train_states
        # take the first state as our starts
        # take the entire sequences as the ground truth
        if definite_start_state is None:
            idxs = np.random.choice(train_states.shape[0], batch_size)
            exp_sample = train_states[idxs].copy()
            exp_samples.append(exp_sample)
            starts = Variable(torch.from_numpy(exp_sample[:, 0:1, :].copy()))
            if opt.cuda:
                starts = starts.cuda()
        else:
            starts = definite_start_state
        
        # sampling

        sample = model.sample(batch_size, g_sequence_len, starts)[0].cpu().data.numpy()
        samples.append(sample)
    if not return_start_states and definite_start_state is not None:
        return np.vstack(samples)
    elif not return_start_states:
        return np.vstack(samples), np.vstack(exp_samples)
    else:
        return np.vstack(samples), np.vstack(exp_samples), starts

def train_epoch(model, data_iter, criterion, optimizer, generator=True):
    total_loss = []
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2,  desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
 
        if generator:
            prob_logs = model.get_log_prob(data, target)
            loss = -prob_logs.mean()
            total_loss.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # note here target is label
            prob = discriminator(data)
            prob = prob[:, 1]
            loss = criterion(prob, target)
            total_loss.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    data_iter.reset()
    return np.mean(total_loss)

def eval_epoch(model, data_iter, criterion, generator=True):
    total_loss = []
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2,  desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
 
        if generator:
            prob_logs = model.get_log_prob(data, target)
            loss = -prob_logs.mean()
            total_loss.append(loss.data[0])
        else:
            # note here target is label
            prob = discriminator(data)
            prob = prob[:, 1]
            loss = criterion(prob, target)
            total_loss.append(loss.data[0])
            
    data_iter.reset()
    return np.mean(total_loss)

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, reward):
        """
        Args:
            prob: (N), torch Variable 
            reward : (N, ), torch Variable
        """
        loss = prob * reward
        loss =  -torch.mean(loss)
        return loss

def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret

def load_expert_data(num):
    train_addr = "anon/train/"
    addrs = os.listdir(train_addr)
    Data = []
    Actions = []
    for d in addrs:
        num -= 1
        if num < 0:
            break
        
        seq_len = int(d.split('-')[2][:2])
        if seq_len != 70:
            continue
        content = open(train_addr + d, 'rb').read()
        data = np.zeros((seq_len, 22), dtype=np.float)
        # print data.shape
        action = np.zeros((seq_len-1, 22), dtype=np.float)
        for i in range(seq_len):
            pre_data = np.asarray(struct.unpack('16i', content[64*i:64*i+64]), dtype=np.float)
            for j in range(11):
                data[i, 2*j] = np.clip(pre_data[j] / 360, 0, 399) / 400
                data[i, 2*j+1] = np.clip(pre_data[j] % 360, 0, 359) / 360
        
            if i > 0:
                action[i-1] = data[i] - data[i-1]
        
        # print data.shape
        data = data[:-1]
        # print data.shape
        Data.append(data)
        Actions.append(action)

    Data = np.stack(Data)
    # print Data.shape
    Actions = np.stack(Actions)
    
    tot_data = Data.shape[0]
    #rand_ind = np.random.permutation(tot_data)
    #Data, Actions = Data[rand_ind], Actions[rand_ind]
    train_data, train_action = Data[:int(tot_data*0.8)], Actions[:int(tot_data*0.8)]
    print(train_data.shape)
    val_data, val_action = Data[int(tot_data*0.8):], Actions[int(tot_data*0.8):]
    
    ave_stepsize = np.mean(np.abs(train_action), axis = (0, 1))
    std_stepsize = np.std(train_action, axis = (0, 1))
    ave_length = np.mean(np.sum(np.sqrt(np.square(train_action[:, :, ::2]) + np.square(train_action[:, :, 1::2])), axis = 1), axis = 0)
    ave_near_bound = np.mean((train_data < 1.0 / 100.0) + (train_data > 99.0 / 100.0), axis = (0, 1))
    print(ave_stepsize, std_stepsize, ave_length, ave_near_bound)
    with open("val_stats.txt", "a") as text_file:
        text_file.write('Expert:\n')
        text_file.write('ave_stepsize: ' + to_string(ave_stepsize) + '\n')
        text_file.write('std_stepsize: ' + to_string(std_stepsize) + '\n')
        text_file.write('ave_length: ' + to_string(ave_length) + '\n')
        text_file.write('ave_near_bound: ' + to_string(ave_near_bound) + '\n')
        text_file.write('\n')
    
    print("train_data.shape:", train_data.shape, "val_data.shape:", val_data.shape)
    return train_data, train_action, val_data, val_action, ave_stepsize, std_stepsize, ave_length, ave_near_bound

if __name__ == "__main__":
    print ("Starting to load to data")
    # Geneating data with target lstm
    target_lstm = Target_LSTM(g_state_dim, g_hidden_dim, g_action_dim, opt.cuda, num_layers=1).double()
    # generate random starts
    train_states, train_actions = target_lstm_generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM)
    
    print ("Done loading data")
    random.seed(SEED)
    np.random.seed(SEED)

    # Define Networks
    generator = Generator(g_state_dim, g_hidden_dim, g_action_dim, opt.cuda, num_layers=1).double()
    discriminator = Discriminator(d_num_class, d_state_dim, d_hidden_dim, num_layers=1).double()

    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        target_lstm = target_lstm.cuda()
    
    # Load data from file
    gen_data_iter = GenDataIter(train_states, train_actions, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.BCELoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.001)

    if opt.cuda:
        gen_criterion = gen_criterion.cuda()

    print('Pretrain with log probs ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        update = None if graph_pretrain_generator is None else 'append'
        graph_pretrain_generator = vis.line(X = np.array([epoch]), Y = np.array([loss]), win = graph_pretrain_generator, update = update, opts=dict(title="pretrain generator training curve"))

    # Pretrain Discriminator
    dis_criterion = nn.BCELoss(size_average=True)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    total_iter = 0
    print ("Pretrain Discriminator ...")
    for epoch in range(2):
        generated_samples, exp_samples = generate_samples(generator, BATCH_SIZE, train_states.shape[0], train_states)
        dis_data_iter = DisDataIter(train_states, generated_samples, BATCH_SIZE)
        for _ in range(10):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer, generator=False)
            print('Epoch [%d], Iter[%d] loss: %f' % (epoch, _, loss))
            update = None if graph_pretrain_discriminator is None else 'append'
            graph_pretrain_discriminator = vis.line(X = np.array([total_iter]), Y = np.array([loss]), win = graph_pretrain_discriminator, update = update, opts=dict(title="pretrain discriminator training curve"))
            total_iter += 1
    
    # Adversarial Training 
    rollout = Rollout(generator, 0.8)
    print ("#####################################################")
    print ("Start Adversarial Training...\n")
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(), lr=0.001)
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()
    gen_criterion = nn.BCELoss(size_average=True)
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    dis_criterion = nn.BCELoss(size_average=True)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    total_iter = 0
    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(3):
            samp_ind = np.random.choice(train_states.shape[0], BATCH_SIZE)
            mod_samples = torch.from_numpy(train_states[samp_ind].copy())
            starts = Variable(mod_samples[:, :1, :].clone())
            if opt.cuda:
                starts = starts.cuda()

            samples, targets = generator.sample(BATCH_SIZE, g_sequence_len, starts)
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            print("ave_rewards = {}".format(np.mean(rewards)))
            rewards = Variable(torch.Tensor(rewards)).contiguous().view((-1,))
            if opt.cuda:
                rewards = torch.exp(rewards.cuda()).contiguous().view((-1,))

            prob = generator.get_log_prob(samples, targets).contiguous().view((-1,)).float()
            loss = gen_gan_loss(prob, rewards)

            update = None if graph_adversarial_training is None else 'append'
            graph_adversarial_training = vis.line(X = np.array([total_batch]), Y = np.array([-loss.data[0]]), win = graph_adversarial_training, update = update, opts=dict(title="adversarial training loss"))

            print ("adversial training loss - generator[%d]: %f" % (total_batch, loss))
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        rollout.update_params()
        
        for _ in range(1):
            generated_samples, exp_samples = generate_samples(generator, BATCH_SIZE, train_states.shape[0], train_states)
            dis_data_iter = DisDataIter(train_states, generated_samples, BATCH_SIZE)
            for _ in range(1):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer, generator=False)
                total_iter += 1
                print ("adversial training loss - discriminator [%d]: %f" % (total_batch, loss))
                update = None if graph_adversarial_training_discriminator is None else 'append'
                graph_adversarial_training_discriminator = vis.line(X = np.array([total_iter]), Y = np.array([loss]), win = graph_adversarial_training_discriminator, update = update, opts=dict(title="adversarial discriminator training loss"))
                
        if total_batch % VAL_FREQ == 0:
            mod_samples, exp_samples = generate_samples(generator, 1, 1, train_states)
            draw_samples(mod_samples, show_image=False, save_image=True, name="GAN_generated_" + str(total_batch))
            draw_samples(exp_samples, show_image=False, save_image=True, name="GAN_expert_" + str(total_batch))








