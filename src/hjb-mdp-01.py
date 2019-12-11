#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:50:24 2019

@author: songqsh

Goal: implement MDP discretized from n-d HJB with Dirichlet data
"""

import numpy as np    

class mdp:
    def __init__(
            self, 
            dim = 1,
            mesh_n = 10, #better to be even number
            lam = 0.0,
            verbose = True
            ):
        self.dim = dim
        self.mesh_n = mesh_n
        self.h = 2.0/mesh_n
        self.rate = dim/(dim+self.h**2*lam)
        self.value = 0
        v_shape = (np.ones(self.dim)*(self.mesh_n+1)).astype(int)        
        self.value = np.zeros(shape = v_shape) #initial
        a_shape = (np.ones(self.dim)*(self.mesh_n*3+1)).astype(int)
        self.action = np.zeros(shape=a_shape) #actions
        
        if verbose:
            print(str(dim)+'-d MDP from HJB')
    
    #convert index to state
    def i2s(self, tup): 
        return (np.array(tup)*self.h-1.0).reshape(self.dim,1) 
    
    #convert index to action
    def i2a(self, tup):
        return (np.array(tup)*self.h-3.0).reshape(self.dim,1)
            
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
        
        
    #value iteration
    #return
    #   state-value with boundary value
    def value_init(self):         
        #boundary value
        v0 = self.value
        it0 = np.nditer(v0, flags=['multi_index'])
        while not it0.finished:
            s = self.i2s(it0.multi_index)
            if self.is_absorbing(s):
                v0[it0.multi_index]=-np.sum(s**2)
            it0.iternext()
                
    #define one step move
    #input:
    #   d-tuple for state
    #   d-tuple for action
    #return: 
    #   2d-list for possible new state indices after one step
    #   2d-list for their corresponding probabilities
    #   float for instant reward
    
    def step(self, s_ind, a_ind):
        s0 = self.i2s(s_ind)
        a0 = self.i2a(a_ind)
        
        ell = lambda x, a: self.dim+2*np.sum(x**2)+ np.sum(a**2)*0.5
        reward = self.h**2*ell(s0, a0)/self.dim
        
        s1_ind = []
        pr = []

        if self.is_absorbing(s0):
            s1_ind.append(s_ind)
            pr.append(1.0)
        else:
            for i in range(self.dim):
                s1_ind_ = np.array(s_ind)
                s1_ind_[i] += 1
                s1_ind_ = tuple(s1_ind_.tolist())
                s1_ind.append(s1_ind_)
            for i in range(self.dim):
                s1_ind_ = np.array(s_ind)
                s1_ind_[i] -= 1
                s1_ind_ = tuple(s1_ind_.tolist())
                s1_ind.append(s1_ind_)
            pr = np.append(1+self.h*a0, 1-self.h*a0)
            pr = pr.tolist()
        
        return s1_ind, pr, reward

    #value iteration
    def value_iter(self):
        self.value_init()
        v0 = self.value
        v1 = v0.copy()
        
        iter_n = 1
        while True:
            it = np.nditer(v0, flags=['multi_index'])
            while not it.finished:
                s0_i = it.multi_index
                s0 = self.i2s(s0_i)
                if not self.is_absorbing(s0):
                    q1 = []
                    it_a = np.nditer(self.action, flags=['multi_index'])
                    while not it_a.finished:
                        a0_i = it_a.multi_index
                        s1_i, pr, rwd = self.step(s0_i, a0_i)
                        rhs = rwd
                        for k in range(2*self.dim):
                            rhs += v0[s1_i[k]]*pr[k]
                        q1.append(rhs)
                        it_a.iternext()
                    v1[it.multi_index] = self.rate*min(q1)
                it.iternext()
            
            if np.sum((v0-v1)**2)<1e-3:
                v0 = v1.copy()
                break
            v0 = v1.copy()
            iter_n += 1
        return iter_n
            
                    
if __name__ == "__main__":
    m = mdp()
    print(m.value_iter())
    print(m.value)
                  
                    
                
        
                
        
    


    
        
        
            
            
        
    

