#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:51:46 2020

@author: songqsh

Section 1.2.2 of 191206HJB.pdf
"multidimensional pde with quadratic solution"

"""
##import time
#import ipdb

###inputs begin

#configurations for PDE
class Pde:
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim    
        self.lam = 0.
    drift = lambda self,s,a: a
    
    run_cost = lambda self,s,a: (
            self.n_dim + sum([s1**2 for s1 in s])*2.0 
            + sum([a1**2 for a1 in a])/2.0
            )
    
    term_cost = lambda self,s: - sum([s1**2 for s1 in s])
    exact_soln = lambda self,s: - sum([s1**2 for s1 in s])


class Mdp(Pde):
    def __init__(self, n_dim = 2, n_mesh = 8):
        super().__init__(n_dim)
        self.n_mesh= n_mesh  
        self.h_mesh = 1./self.n_mesh #mesh size
        self.v_shape = tuple([self.n_mesh + 1]*self.n_dim)


    #input: list of index
    #return: physicial coordinate
    def i2s(self,ix): 
        return [x * self.h_mesh for x in ix]
    
    def is_interior(self,ix):
        return all(map(lambda a: 0<a<self.n_mesh, ix))
        
    #input: lists of index and action
    #return: discount rate, running cost, list of next index, list of probability
    def step(self, ix, a):
        s = self.i2s(ix)
        b = Pde.drift(Pde, s, a)
        
        lam = self.n_dim/(self.n_dim+self.lam*(self.h_mesh**2))
        run_cost_h = self.h_mesh**2*self.run_cost(s,a)/self.n_dim
        
        ix_next = []; pr_next= []
        #cfd
        if self.is_interior(ix):
            for i in range(self.n_dim):
                ix1 = ix.copy(); ix1[i]+=1; ix_next += [ix1,]
                pr1 = (1+2.*self.h_mesh*b[i])/(self.n_dim*2.0) 
                pr_next += [pr1,]
            for i in range(self.n_dim):
                ix1 = ix.copy(); ix1[i]-=1; ix_next += [ix1,]
                pr1 = (1-2.*self.h_mesh*b[i])/(self.n_dim*2.0) 
                pr_next += [pr1,]
        
        return lam, run_cost_h, ix_next, pr_next
    
    #input:
        #ndarray v of v_shape
        #list of index and action
    #return:
        #q_val assuming v is value
    def q_val(self, v, ix, a):
        lam, run_cost_h, ix_next, pr_next = self.step(ix,a)
        out = run_cost_h
        for ix1, pr1 in zip(ix_next, pr_next):
            out+=pr1*v[tuple(ix1)]
        out *= lam
        return out
 
####################
       

import numpy as np
import itertools
from scipy.optimize import minimize

def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)

#product of a list
def product(l):
    out = 1
    for x in l:
        out *= x
    return out


#value iteration    
mdp = Mdp(n_dim=2, n_mesh=8)
v = np.zeros(mdp.v_shape) #init
for ix in deep_iter(*mdp.v_shape):
    if not mdp.is_interior(ix):
        v[ix] = mdp.term_cost(mdp.i2s(ix))

p_shape = tuple(list(mdp.v_shape)+[mdp.n_dim,])
policy = np.zeros(p_shape) #init

tol = 1e-5; max_iter = 1000
for n_iter in range(max_iter):
    v_cp = np.copy(v)
    err = 0.
    for ix in deep_iter(*mdp.v_shape):
        if mdp.is_interior(ix):
            fun = lambda a: mdp.q_val(v, list(ix), policy[ix])
            res = minimize(fun, np.zeros([mdp.n_dim,]))
            v_cp[ix] = res.fun
            policy[ix] = res.x
            err += (v_cp[ix]-v[ix])**2
    v = np.copy(v_cp)
    if err<tol:
        break
    
print('>>>>', err)
    
#check
err = 0
for ix in deep_iter(*mdp.v_shape):
    exact_val = mdp.exact_soln(mdp.i2s(list(ix)))
    #print(exact_val, v[ix])
    err += (exact_val- v[ix])**2
err = err/product(mdp.v_shape)
    
print('>>>>', err, n_iter)
    
    
        
            
        
    




                    
'''
#### test Mdp
s =[.5,0.6]; a = [0.,0.1];ix1 =[4,7];ix2= [5,9]
m = Mdp(n_mesh=8)
print('test Mdp')
print(m.n_dim, m.n_mesh, m.h_mesh, m.v_shape)
print(m.i2s(ix1), m.i2s(ix2))
print(m.is_interior(ix1), m.is_interior(ix2))
print(m.step_cfd(ix1, a))
print(m.step_cfd(ix2, a))
print(m.run_cost(s,a))
print(m.term_cost(s))
#### test ends

'''
