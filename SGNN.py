#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class SGNN(Module):
    def __init__(self, hidden_size, step=1):
        super(SGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # self.linear_graph = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.attn1 = nn.MultiheadAttention(self.hidden_size, num_heads=1)
        self.attn2 = nn.MultiheadAttention(self.hidden_size, num_heads=1)
        self.highway= nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def SGNNCell(self, A, hidden, mask, alias_inputs, g_emb):
        # g_emb = torch.mean(hidden,dim = 1).unsqueeze(1)
        # get = lambda i: hidden[i][alias_inputs[i]]
        # seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        # ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        # q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
        # alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # g_emb = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1).unsqueeze(1)

        # attn_output,attn_weights = self.attn(g_emb.transpose(0,1), hidden.transpose(0,1), hidden.transpose(0,1))
        # g_emb = torch.mean(attn_output.transpose(0,1),dim=1).unsqueeze(1)
        # print("attn_output",attn_output.shape,hidden.shape)
        # print("attn_weights",attn_weights.shape, g_emb_new.shape)

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # input_in = input_in + self.linear_graph(g_emb)
        # input_out = input_out + self.linear_graph(g_emb)
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        attn_output, attn_weights = self.attn1(hy.transpose(0, 1), g_emb.transpose(0, 1), g_emb.transpose(0, 1))
        #### write (1 - alpha)
        # hy_new = hy + torch.matmul(attn_weights, g_emb)
        hy_new = hy + attn_weights*g_emb - attn_weights*hy
        star_update_output,star_update_weights = self.attn2(g_emb.transpose(0, 1), hy_new.transpose(0, 1), hy_new.transpose(0, 1))
        # m = nn.Softmax(dim=1)
        # beta = m(star_update_weights)
        # print(g_emb.shape)
        # print(star_update_output.shape)
        # g_emb = torch.matmul(beta,hy_new)
        return hy_new, star_update_output.transpose(0,1)

    def highway_networks(self,hidden_orig,hidden_final):
        temp = self.highway(torch.cat([hidden_orig,hidden_final],2))
        gate = torch.sigmoid(temp)
        return(hidden_final + gate*hidden_orig - gate*hidden_final)
        # print(temp.shape)
        # return(torch.sigmoid(temp))

    def forward(self, A, hidden, mask, alias_inputs, g_emb):
        hidden_orig = hidden
        for i in range(self.step):
            hidden,g_emb = self.SGNNCell(A, hidden, mask, alias_inputs, g_emb)
        # print(hidden.shape)
        # print(hidden_orig.shape)
        # print(torch.cat([hidden_orig,hidden],1).shape)
        return self.highway_networks(hidden_orig,hidden), g_emb


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = SGNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_star = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.position_encoding = Parameter(torch.Tensor(1000, self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, g_emb):
        # print(mask.shape)
        # print(hidden.shape)
        # print("lol")
        hidden = hidden + self.position_encoding[hidden.shape[1]-1][hidden.shape[2]-1]
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        # ht = ht + self.position_encoding[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        q3 = self.linear_star(g_emb)
        alpha = self.linear_three(torch.sigmoid(q1 + q2 + q3))
        # print(alpha.shape)
        # print("hidden_shape",hidden.shape)
        # print("mask_shape", mask.shape)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        ###### Add layer normalization here
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A, mask, alias_inputs):
        hidden = self.embedding(inputs)
        g_emb = torch.mean(hidden, dim=1).unsqueeze(1)
        hidden, g_emb = self.gnn(A, hidden, mask, alias_inputs, g_emb)
        return hidden,g_emb


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    # print("mask:", mask.shape)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden,g_emb = model(items, A, mask, alias_inputs)
    # print("items:" ,items.shape)
    # print("hidden:" ,hidden.shape)
    # print("mask:", mask.shape)
    # print("alias_inputs:", alias_inputs.shape)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # print("seq_hideen:", seq_hidden.shape)
    return targets, model.compute_scores(seq_hidden, mask,g_emb)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
