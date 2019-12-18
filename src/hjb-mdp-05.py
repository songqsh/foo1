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
    
    #domain is a unit hyper cube        
    def is_interior(self, s):
        return all(0<s<1)

p = Pde()
    
#cfd2mdp
class Mdp:
    def __init__(self, pde = p, n_mesh_s = 8, n_mesh_a = 16, method='cfd'):
        self.pde = p
        self.n_mesh_s = n_mesh_s
        self.n_mesh_a = n_mesh_a
        self.method = method
        self.dim = pde.dim
        
        ####domain of mdp
        h_s = self.pde.limit_s/n_mesh_s #mesh size in state
        h_a = self.pde.limit_a/n_mesh_a #mesh size in action
        v_shape = tuple([n_mesh_s + 1]*self.dim)
        a_shape = tuple([n_mesh_a + 1]*self.dim)
        
        def is_interior(*ix_s):
            return all([0<x<n_mesh_s for x in ix_s])
        
        out.update({
                'v_shape': v_shape,
                'a_shape': a_shape,
                'is_interior': is_interior
                })
        ####domain
 
       # convert index(tuple) to state
        def i2s(*ix): 
            return np.array([x * h_s for x in ix])       
        out['i2s'] = i2s
        #convert index to action
        def i2a(*ix):
            return np.array([x * h_a for x in ix])
        out['i2a'] = i2a


       
        ########running and terminal costs
        def run_cost(ix_s,ix_a):
            return self.run_cost(i2s(*ix_s), i2a(*ix_a))*h_s**2/self.dim
        
        def term_cost(ix_s):
            return self.term_cost(i2s(*ix_s))
        out.update({
                'run_cost': run_cost,
                'term_cost': term_cost
                })
        #########
        
        #####transition
        #return:
        #   a list of nbd indices
        #   a list of prob
        def step(ix_s, ix_a):
            ix_next_s_up = (np.array(ix_s)+np.eye(self.dim)).astype(int).tolist()
            ix_next_s_dn = (np.array(ix_s)-np.eye(self.dim)).astype(int).tolist()
            ix_next_s = [tuple(ix) for ix in ix_next_s_up+ix_next_s_dn]
            
            pr=[]
            if method == 'cfd':
                b = self.drift(i2s(*ix_s), i2a(*ix_a))
                pr_up = ((1+h_s*b)/self.dim).tolist()
                pr_dn = ((1-h_s*b)/self.dim).tolist()
                pr = pr_up+pr_dn
            
            return ix_next_s, pr
        out.update({'step': step})
    
        return out
    
