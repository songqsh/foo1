# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, m=1):
        super(Net, self).__init__()
        # input/output dim
        in_dim = 1
        out_dim = 1
        
        self.m = m
        
        # kernel
        H = 10
        self.fc1 = nn.Linear(in_dim, H)
        self.fc2 = nn.Linear(H, out_dim)
        
    def forward(self, x):
        #layers
        if self.m == 1:
            x = self.fc1(x) 
            x = torch.sigmoid(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            
        return x
    
    
#loss function 
criterion = nn.MSELoss()

# target function
f = lambda x: np.abs(x)+1




