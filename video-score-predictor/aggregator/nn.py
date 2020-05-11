import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

class Ensemble(nn.Module):

    def __init__(self, mode='mean'):
        self.predictors = []
        if mode == 'min':
            self.aggregation = min_aggregation
        elif mode == 'mean':
            self.aggregation = mean_aggregation
        elif mode == 'max':
            self.aggregation = max_aggregation
        else:
            raise Exception('Unknown aggregation mode: ' + mode) 

    def forward(self, x):
        return self.aggregation(torch.tensor([predictor(x) for predictor in self.predictors]))

    def add_predictor(self, predictor):
        self.predictors.append(predictor)            

class UninormAggregator(nn.Module):
    
    def __init__(self, num_params, tnorm="lukasiewicz", normalize_neutral=False, init_neutral=0., off_diagonal='min'):
        super(UninormAggregator, self).__init__()    
        self.num_params = num_params           
        self.neutral = nn.Parameter(torch.ones(num_params) * init_neutral) # tconorm is default       
        if tnorm == "lukasiewicz":
            self.tnorm = lukasiewicz_tnorm
            self.tconorm = lukasiewicz_tconorm
        elif tnorm == "product":
            self.tnorm = product_tnorm
            self.tconorm = product_tconorm
        else:
            raise Exception('Unknown tnorm: ' + tnorm)   
        if off_diagonal == 'min':
            self.off_diagonal_aggregation = min_aggregation
        elif off_diagonal == 'mean':
            self.off_diagonal_aggregation = mean_aggregation
        elif off_diagonal == 'max':
            self.off_diagonal_aggregation = max_aggregation
        else:
            raise Exception('Unknown off-diagonal aggregator: ' + off_diagonal)   
        self.normalize_neutral = normalize_neutral
        self.fc = (lambda x : x)        

    def activate_linear(self):
        self.fc = nn.Linear(self.num_params, self.num_params)

    def print_parameters(self):
        for p in self.parameters():
            print(p.name,p.data,p.requires_grad)

    def print_gradient(self):
        print(self.neutral.grad)

    def init_params(self, params):
        self.neutral = nn.Parameter(params)    

    def forward(self, x):
        if self.normalize_neutral:
            return self.fc(self.uninorm(x, self.neutral / x.shape[1]))    
        else:      
            return self.fc(self.uninorm(x, self.neutral))

    def uninorm(self, x, neutral):     
        if x.shape[1] == 1:
            return x[:,0]
        if x.shape[1] == 2:            
            return self.min_uninorm(x, neutral)
        half = x.shape[1] // 2
        return self.uninorm(torch.stack((self.uninorm(x[:,:half], neutral),self.uninorm(x[:,half:], neutral))).t(), neutral)

    def min_uninorm(self, x, neutral):    
        y = torch.zeros(x.shape[0], dtype=torch.float32)  
        mask_00 = np.logical_and(x[:,0] <= neutral, x[:,1] <= neutral).bool()  
        if True in mask_00:
            neutral_00 = neutral[mask_00]   
            neutral_00_full = neutral_00.repeat(x.shape[1],1).t()     
            y[mask_00] = neutral_00 * self.tnorm(torch.div(x[mask_00], neutral_00_full))
        mask_11 = np.logical_and(x[:,0] > neutral, x[:,1] > neutral).bool()  
        if True in mask_11: 
            neutral_11 = neutral[mask_11]
            neutral_11_full = neutral_11.repeat(x.shape[1],1).t()
            y[mask_11] = neutral_11 + (1 - neutral_11) * self.tconorm(torch.div(x[mask_11] - neutral_11_full,1 - neutral_11_full))
        mask_xx = np.logical_xor(x[:,0] <= neutral, x[:,1] <= neutral).bool()             
        if True in mask_xx:       
            y[mask_xx] = self.off_diagonal_aggregation(x[mask_xx])
        return y

    def clamp_params(self):
        self.neutral.data.clamp_(0.,1.)

class CovidNoCovidNet(nn.Module):

    def __init__(self, num_params, tnorm="lukasiewicz", normalize_neutral=False, init_neutral=0., off_diagonal='min'):
        super(CovidNoCovidNet, self).__init__()

        self.local_aggregator = UninormAggregator(num_params-1, tnorm, normalize_neutral, init_neutral, off_diagonal, True)
        self.global_aggregator = UninormAggregator(2, tnorm, normalize_neutral, init_neutral, off_diagonal)

    def forward(self, x):
        x_neg = x[0,:]
        x_pos = self.local_aggregator(x[1:,:].t())
        return self.global_aggregator(torch.stack((x_neg, x_pos)))

    def print_parameters(self):
        for p in self.parameters():
            print(p.name,p.data,p.requires_grad)

    def print_gradient(self):
        self.local_aggregator.print_gradient()
        self.global_aggregator.print_gradient()

    def clamp_params(self):
        self.local_aggregator.clamp_params()
        self.global_aggregator.clamp_params()

class ScoreHierarchyNet(nn.Module):

    def __init__(self, num_params, tnorm="lukasiewicz", normalize_neutral=False, init_neutral=0., off_diagonal='min'):
        super(ScoreHierarchyNet, self).__init__()    

        self.score_aggregator = UninormAggregator(num_params-1, tnorm, normalize_neutral, init_neutral, off_diagonal)

    def forward(self, x):
        pos_neg = F.softmax(torch.stack((torch.mean(x[0,:]),torch.mean(torch.sum(x[1:,:],dim=0)))),dim=0)   
        output = torch.zeros(x.shape[0])   
        if pos_neg[0] >= 0.5:
            output[0] = pos_neg[0]
            output[1:] = pos_neg[1] * F.softmax(self.score_aggregator(x[1:,:]),dim=0)
        else:
            output[1:] = self.score_aggregator(x[1:,:])
        return output

    def print_parameters(self):
        for p in self.parameters():
            print(p.name,p.data,p.requires_grad)

    def print_gradient(self):
        self.score_aggregator.print_gradient()

    def clamp_params(self):
        self.score_aggregator.clamp_params()


def lukasiewicz_tnorm(x):
    return torch.max(torch.zeros(x.shape[0]), torch.sum(x,1)-1) 

def lukasiewicz_tconorm(x):
    return torch.min(torch.ones(x.shape[0]),torch.sum(x,1)) 

def product_tnorm(x):
    return torch.prod(x,dim=1)

def product_tconorm(x):
    return torch.sum(x,dim=1) - torch.prod(x,dim=1)

def min_aggregation(x):
    return torch.min(x, dim=1).values 

def mean_aggregation(x):
    return torch.mean(x, dim=1)

def max_aggregation(x):
    return torch.max(x, dim=1).values 

