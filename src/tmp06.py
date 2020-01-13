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
    def __init__(self):
        self.n_dim = 2    
        self.lam = 0
    drift = lambda self,s,a: a
    
    run_cost = lambda self,s,a: (
            self.n_dim + sum([s1**2 for s1 in s])*2.0 
            + sum([a1**2 for a1 in a])/2.0
            )
    
    term_cost = lambda self,s: - sum([s1**2 for s1 in s])
    
#### test Pde
print('test begins')
p = Pde()
s =[.5, .3]; a = [.2, 0.8]
print(p.drift(s,a), p.run_cost(s,a), p.term_cost(s))
print('test ends')
####test end

class Mdp(Pde):
    def __init__(self, n_mesh = 8):
        super().__init__()
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
    def step_cfd(self, ix, a):
        s = self.i2s(ix)
        b = Pde.drift(Pde, s, a)
        
        lam = self.n_dim/(self.n_dim+self.lam*(self.h_mesh**2))
        run_cost_h = self.h_mesh**2*self.run_cost(s,a)/self.n_dim
        #ix_list = list(ix)
        
        ix_next = []; pr_next= []
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
    
                    


    
#### test Mdp
print('test Mdp')
m = Mdp()
print(m.n_dim, m.n_mesh, m.h_mesh, m.v_shape)
ix1 =[4, 9]
ix2= [5, 2]
print(m.i2s(ix1), m.i2s(ix2))
print(m.is_interior(ix1), m.is_interior(ix2))
print(m.step_cfd(ix1, a))
print(m.step_cfd(ix2, a))
#### test ends


