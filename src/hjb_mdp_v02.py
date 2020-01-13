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


class Mdp:
    def __init__(self, n_mesh = 8, pde = p):
        self.pde = p
        self.n_dim = p.n_dim
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
        b = self.pde.drift(s, a)
        
        lam = self.n_dim/(self.n_dim+self.pde.lam*(self.h_mesh**2))
        run_cost_h = self.h_mesh**2*self.pde.run_cost(s,a)/self.n_dim
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
    
                    
    


