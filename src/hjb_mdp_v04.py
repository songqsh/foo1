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
        print('>>> n_dim: '+str(n_dim))
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
        print('>>> n_mesh: '+str(n_mesh))


    #input: list of index
    #return: physicial coordinate
    def i2s(self,ix): 
        return [x * self.h_mesh for x in ix]
    
    def is_interior(self,ix):
        return all(map(lambda a: 0<a<self.n_mesh, ix))
        
    #input: lists of index and action
    #return: discount rate, running cost, list of next index, list of probability
    def step(self, ix, a, fd='cfd'):
        s = self.i2s(ix)
        b = Pde.drift(Pde, s, a)
        if fd=='cfd':
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
        elif fd=='ufd':
            c = self.n_dim+sum([abs(b1) for b1 in b])*self.h_mesh
            b_plus = [(abs(b1)+b1)/2. for b1 in b]
            b_minus = [(abs(b1)-b1)/2. for b1 in b]
            lam = c/(c+self.h_mesh**2*self.lam)
            run_cost_h = self.h_mesh**2*self.run_cost(s,a)/c
            ix_next = []; pr_next= []
            #ufd
            if self.is_interior(ix):
                for i in range(self.n_dim):
                    ix1 = ix.copy(); ix1[i]+=1; ix_next += [ix1,]
                    pr1 = (1+2.*self.h_mesh*b_plus[i])/(c*2.0) 
                    pr_next += [pr1,]
                for i in range(self.n_dim):
                    ix1 = ix.copy(); ix1[i]-=1; ix_next += [ix1,]
                    pr1 = (1-2.*self.h_mesh*b_minus[i])/(c*2.0) 
                    pr_next += [pr1,]        
        return lam, run_cost_h, ix_next, pr_next
    

 
####################
import itertools

def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)


#product of a list
def product(l):
    out = 1
    for x in l:
        out *= x
    return out
############ Your solver

import numpy as np

class ValueIter(Mdp):
    def __init__(self, n_dim = 1, n_mesh = 8, fd = 'ufd'):
        super().__init__(n_dim, n_mesh)
        self.fd = fd
        #value iteration init
        self.v = np.zeros(self.v_shape) #init
        self.a_space = tuple([3*n_mesh+1,]*n_dim)
        self.p_shape = tuple(list(self.v_shape)+[self.n_dim,])
        self.policy = np.zeros(self.p_shape) #init
        self.tol = 1e-5; self.max_iter = 1000
        
        #boundary value
        for ix in deep_iter(*self.v_shape):
            if not self.is_interior(ix):
                self.v[ix] = self.term_cost(self.i2s(ix))
    
    
    def i2a(self, ix):
        return [ix1*1./self.n_mesh for ix1 in ix]
    
    #input:
        #list of index and action, method
    #return:
        #q_val assuming v is value
    def q_val(self, ix, a, fd):
        lam, run_cost_h, ix_next, pr_next = self.step(ix,a,fd)
        out = run_cost_h
        for ix1, pr1 in zip(ix_next, pr_next):
            out+=pr1*self.v[tuple(ix1)]
        out *= lam
        return out
    
    #minimum over action space
    def min_a(self, fun):
        out_ind = [0,]*self.n_dim; out_val = fun(self.i2a(out_ind))
        for ix in deep_iter(*self.a_space):
            if fun(self.i2a(ix))<out_val:
                out_ind = ix; out_val = fun(self.i2a(ix))
        return out_ind, out_val

    def solver(self):    
        for n_iter in range(self.max_iter):
            v_cp = np.copy(self.v)
            err = 0.
            for ix in deep_iter(*self.v_shape):
                if self.is_interior(ix):
                    fun = lambda a: self.q_val(list(ix), a, self.fd)
                    out_ix, out_v = self.min_a(fun)
                    self.policy[ix] = list(out_ix); v_cp[ix] = out_v
                    err += (v_cp[ix]-self.v[ix])**2
            self.v = np.copy(v_cp)
            if err<self.tol:
                break
        return err, n_iter
    


#check
vi = ValueIter(n_dim=2, n_mesh=8, fd='ufd')
err, n_iter = vi.solver()
print('>>>running err:' +str(err)+' n_iter: '+ str(n_iter))

err = 0
exact_val = np.zeros(vi.v_shape)

for ix in deep_iter(*vi.v_shape):
    exact_val[ix] = vi.exact_soln(vi.i2s(list(ix)))
    err += (exact_val[ix]- vi.v[ix])**2
err = err/product(vi.v_shape)
    
print('>>>>err:'+str(err)+',n_iter: '+str(n_iter))
    
    
import matplotlib.pyplot as plt    

if vi.n_dim==1:
    x_cod = np.zeros(vi.v_shape)
    for ix in deep_iter(*vi.v_shape):
        x_cod[ix] = vi.i2s(list(ix))[0]
        
    plt.plot(x_cod, vi.v, x_cod, exact_val)
                
        




