#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:31:01 2019

@author: songqsh
"""
import numpy as np
dim = 2
mesh_n = 4
h = 2/mesh_n
x_shape = (np.ones(dim)*(mesh_n+1)).astype(int)
x = np.zeros(shape=x_shape)

def is_absorbing(state):
    if state.shape != (dim, 1):
        print('warning: state dimension is not right')
        return False
    elif np.max(np.abs(state)) < 1:
        return False
    else:
        return True
    
i2s = lambda i: (np.array(i)*h -1.0).reshape(dim,1)

itx = np.nditer(x, flags=['multi_index'])
while not itx.finished:
    #s = (np.array(itx.multi_index)*h -1.0).reshape(dim,1)
    s = i2s(itx.multi_index)
    if is_absorbing(s):
        x[itx.multi_index]=-np.sum(s**2)
    itx.iternext()
