#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Dec 10 10:50:24 2019

@author: songqsh

Goal: implement MDP discretized from n-d HJB with Dirichlet data
see probelm here:
    https://github.com/songqsh/foo1/blob/master/doc/191206HJB.pdf
"""


import numpy as np
#import time
#import pdb

#alternative for nditer, but for both tensor and ndarray
def deep_iter(data, ix=tuple()):
    try:
        for i, element in enumerate(data):
            yield from deep_iter(element, ix + (i,))
    except:
        yield ix, data
        
        

class Pde:
    def __init__(
            self,
            dim=1,
            lam=0.0,
            drift = lambda s,a: a,
            run_cost = lambda s,a: len(s) + np.sum(s**2)*2.+ np.sum(a**2)/2.0,
            term_cost = lambda s, a: -np.sum(s**2),
            limit_s = 1.0, #l-infinity limit for state
            limit_a = 2.0, #l-infinity limit for action
            verbose=True
    ):
        self.dim = dim
        self.lam = lam
        self.drift = drift
        self.run_cost = run_cost
        self.term_cost = term_cost            
        self.limit_s = limit_s
        self.limit_a = limit_a

        if verbose:
            print(str(dim) + '-dim HJB')
    
    #domain is a hyper cube        
    def is_interior(self, s):
        return all(np.abs(s)<1)
    
    #cfd2mdp
    def mdp(self, n_mesh_s = 8, method='cfd'):
        out = {}
        out['n_mesh_s'] = n_mesh_s
        h_s = 2*self.limit_s/n_mesh_s #mesh size in state

        out['n_mesh_a'] = int(n_mesh_s*self.limit_a/self.limit_s)
        h_a = 2*self.limit_a/out['n_mesh_a']

        
        # convert index(tuple) to state
        def i2s(*ix): 
            return np.array([x * h_s - self.limit_s for x in ix])       
        out['i2s'] = i2s
        
        def i2a(*ix):
            return np.array([x * h_a - self.limit_a  for x in ix])
        out['i2a'] = i2a
        
        
        return out
    
