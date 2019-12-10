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
            verbose = True
            ):
        self.dim = dim
        if verbose:
            print(str(dim)+
                  '-d MDP from HJB'
                  )
            
    #define reflecting states
    #input: n-d array for a state
    #return: true/false
    def is_reflecting(self, state):
        return False
    
    #define absorbing states
    #input: n-d array for a state
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
    #   n-d array for state
    #   n-d array for action
    #return: 
    #   n-d array for state after one step 
    #   a float number for reward
    
        
        
            
            
        
    

