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
import time
#import ipdb

import itertools
def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)
        
        

class Pde:
    def __init__(
            self,
            dim=1,
            lam=0.0,
            drift = lambda s,a: a,
            run_cost = lambda s,a: len(s) + np.sum(s**2)*2.+ np.sum(a**2)/2.0,
            term_cost = lambda s: -np.sum(s**2),
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
    
    #cfd2mdp
    def mdp(self, n_mesh_s = 8, n_mesh_a = 16, method='cfd'):
        out = {}
        
        ####domain of mdp
        h_s = self.limit_s/n_mesh_s #mesh size in state
        h_a = self.limit_a/n_mesh_a #mesh size in action
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
        #out['i2a'] = i2a


       
        ########running and terminal costs and discount rate
        def run_cost(ix_s,ix_a):
            return self.run_cost(i2s(*ix_s), i2a(*ix_a))*h_s**2/self.dim
        
        def term_cost(ix_s):
            return self.term_cost(i2s(*ix_s))
        
        rate = self.dim/(self.dim+self.lam*(h_s**2))
        out.update({
                'run_cost': run_cost,
                'term_cost': term_cost,
                'rate': rate
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
                pr_up = ((1+h_s*b)/self.dim/2.0).tolist()
                pr_dn = ((1-h_s*b)/self.dim/2.0).tolist()
                pr = pr_up+pr_dn
            
            return ix_next_s, pr
        out.update({'step': step})
    
        return out
    


def value_iter(v_shape, a_shape, i2s, is_interior, 
               run_cost, term_cost, rate, step):
    dim = len(v_shape)
    v0 = np.zeros(v_shape)
    a = np.zeros(a_shape)
    
    # boundary value
    for ix_s in deep_iter(*v_shape):
        if not is_interior(*ix_s):
            v0[ix_s]=term_cost(ix_s)
    v1 = v0.copy()



    for iter_n in range(100):
        for ix_s0 in deep_iter(*v_shape):
            if is_interior(*ix_s0):
                q1 = []
                for ix_a in deep_iter(*a_shape):
                    rhs = run_cost(ix_s0, ix_a)
                    ix_s1, pr = step(ix_s0, ix_a); 
                    for k in range(2*dim):
                        rhs += v0[ix_s1[k]]*pr[k]
                    q1 += [rhs,]
                v1[ix_s0] = rate*min(q1); 
                

        if np.max(np.abs(v0 - v1)) < 1e-3:
            v0 = v1.copy()
            break
        v0 = v1.copy();  
               
        #iter_n += 1
    return iter_n, v0

if __name__=="__main__":
    p = Pde(dim=1); m = p.mdp(n_mesh_s=8)
    start_time = time.time()
    n, v = value_iter(**m)
    end_time = time.time()
    print('>>>time elapsed is: ' + str(end_time - start_time))

    def true_soln(s):
        return -np.sum(s**2)
    err = []
    for ix_s in deep_iter(*m['v_shape']):
        err0 = np.abs(v[ix_s] - true_soln(m['i2s'](*ix_s)))
        err += [err0, ]
    print('>>> sup norm error is: ' + str(max(err)))
    print('>>> number of iterations is: ' + str(n))
    




