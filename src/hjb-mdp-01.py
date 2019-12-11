#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:50:24 2019

@author: songqsh

Goal: implement MDP discretized from n-d HJB with Dirichlet data
"""

import numpy as np

class MDP:
    def __init__(
            self, 
            dim = 2,
            mesh_n = 20, #better to be even number
            lam = 0.0,
            verbose = True
            ):
        self.dim = dim
        self.h = 2.0/mesh_n
        self.rate = dim/(dim+self.h**2*lam)
        if verbose:
            print(str(dim)+
                  '-d MDP from HJB'
                  )
            
    #define reflecting states
    #input: d-array for a state
    #return: true/false
    def is_reflecting(self, state):
        return False
    
    #define absorbing states
    #input: d-array for a state
    #return: true/false
    def is_absorbing(self, state):
        if state.shape != (self.dim, 1):
            print('warning: state dimension is not right')
            return False
        elif np.max(np.abs(state)) < 1:
            return False
        else:
            return True
        
    #define one step move
    #input:
    #   d-array for state
    #   d-array for action
    #return: 
    #   d*2d-array for possible states after one step
    #   1*d-array for their transition probability
    #   float for instant reward
    
    def step(self, state, action):
        del_state = np.append(np.eye(self.dim), - np.eye(self.dim), axis =1)*self.h
        if self.is_absorbing(state):
            next_state = state
        else:
            next_state = state+del_state
            
        trans_prob = np.append(1+self.h*action, 1-self.h*action).reshape(1, 2*self.dim)
        
        ell = lambda x, a: self.dim+2*np.sum(x**2)+ np.sum(a**2)*0.5
        reward = self.h**2*ell(state, action)/self.dim
            
        return next_state, trans_prob, reward
    

    
        
        
            
            
        
    

