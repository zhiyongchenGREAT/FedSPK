#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from utils import accuracy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		
		super(LossFunction, self).__init__()
		self.test_normalize = True
	    
		self.criterion  = torch.nn.CrossEntropyLoss()
		self.fc 		= nn.Linear(nOut,nClasses)

		print('Initialised Softmax Loss')
		
	def forward(self, x, label=None):
		x 		= self.fc(x)
		nloss   = self.criterion(x, label)
		prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1

class LossFunction_with_transformer(nn.Module):
    def __init__(self, nOut, nClasses, **kwargs):
        
        super(LossFunction_with_transformer, self).__init__()
        self.test_normalize = True
        
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.fc 		= nn.Linear(nOut,nClasses)

        self.encoder_layers = TransformerEncoderLayer(d_model=192, nhead=2, dim_feedforward=192, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=2)

        print('Initialised Softmax Loss')
        
    def forward(self, x, label=None):

        trans_out = self.transformer_encoder(x)

        if label == None:
            return trans_out

        x = self.fc(trans_out)
        
        nloss  = self.criterion(x, label)
        prec1	= accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
