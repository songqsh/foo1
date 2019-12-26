#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:11:52 2019

@author: songqsh
"""

#import time
#import ipdb
'''
import torch
import torch.nn as nn
from torch.autograd import grad
'''

import itertools
def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)
        

####paras for pde
n_dim_ = 2
lam_ = 0.
def drift(s):
    return [0.]*n_dim_
def run(s):
    return sum(map(lambda a: a**2, s))*(lam_- n_dim_)
def term(s):
    return sum(map(lambda a: a**2, s))
def is_interior(s):  #domain 
    return all(map(lambda a: 0.<a<1., s))
######paras for computation
n_mesh_ = 8

###### index domain
v_shape_ = tuple([n_mesh_ + 1]*n_dim_)
h_ = 1./n_mesh_
def i2s(ix): 
    return [x * h_ for x in ix]

#####transition
#return:
#   a list of next indices
#   a list of prob
def step(ix, method='cfd'):
    s = i2s(ix)
    b = drift(s)
    ix_list = list(ix)
    
    ix_next = []; pr_next= []
    if is_interior(s) and method=='cfd':
        for i in range(n_dim_):
            ix1 = ix_list; ix1[i]+=1; ix_next += [tuple(ix1),]
            pr1 = (1+h_*b[i])/n_dim_/2.0; pr_next += [pr1,]
        for i in range(n_dim_):
            ix1 = ix_list; ix1[i]-=1; ix_next += [tuple(ix1),]
            pr1 = (1-h_*b[i])/n_dim_/2.0; pr_next += [pr1,]
    
    return ix_next, pr_next


########running and terminal costs and discount rate
def run_h(ix_s):
    return run(i2s(ix_s))*h_**2/n_dim_
def term_h(ix_s):
    return term(i2s(ix_s))
discount_rate = n_dim_/(n_dim_+lam_*(h_**2))

#inital value
def value(s):
    return sum(map(lambda a: a*2-1, s))
    

####Bellma equation
def bellman(ix):
    s = i2s(ix)
    lhs = value(s); rhs = 0
    if is_interior(s):
        rhs +=run_h(ix)
        ix_next, pr_next = step(ix)
        for ix1, pr1 in zip(ix_next, pr_next):
            rhs += pr1*value(i2s(ix1))
        rhs *= discount_rate
    else:
        rhs = term_h(ix)
    return (rhs - lhs)**2
        
            
        
        

