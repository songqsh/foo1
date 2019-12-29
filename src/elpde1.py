#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:26:05 2019

@author: songqsh
"""
#import ipdb
import time
import torch
import torch.nn as nn
import itertools
def deep_iter(*shape):
    iters = (range(i) for i in shape)
    return itertools.product(*iters)



class Pde:
    def __init__(
            self,
            n_dim_ = 2,
            lam_ = 0.,
            verbose=True
    ):
        self.n_dim_ = n_dim_
        self.lam_ = lam_
        if verbose:
            print('>>>>Elliptic Linear PDE with '+str(n_dim_) + '-dim')
    
def drift(self, s):
    return [0.]*self.n_dim_
Pde.drift = drift

def run(self, s):
    return float(-self.n_dim_)
Pde.run = run

def term(self, s):
    return sum(map(lambda a: (a-.5)**2, s))
Pde.term = term

def is_interior(self, s):  #domain 
    return all(map(lambda a: 0.<a<1., s))    
Pde.is_interior = is_interior

def exact_soln(self, s):
    return sum(map(lambda a: (a-.5)**2, s))
Pde.exact_soln= exact_soln



###########MDP
class Mdp:
    def __init__(
            self,
            pde,
            n_mesh_ = 8, 
            method ='cfd'
            ):
            
        ###### index domain
        self.pde = pde
        self.n_mesh_ = n_mesh_
        self.method = method
        
        self.n_dim_ = pde.n_dim_
        self.v_shape_ = tuple([n_mesh_ + 1]*self.n_dim_)
        self.v_size_ = (n_mesh_+1)**self.n_dim_
        self.h_ = 1./n_mesh_
        self.discount_rate = (
                self.n_dim_/(self.n_dim_+self.pde.lam_*(self.h_**2))
                )
        print(
                '>>>>MDP with ' + str(self.n_dim_)
                + '-dim ' + str(self.n_mesh_) + ' mesh num'
                )
    
    def i2s(self, ix): 
        return [x * self.h_ for x in ix]
    
    def is_interior(self, ix):
        return all(map(lambda a: 0<a<self.n_mesh_, ix))
    
    #####transition
    #return:
    #   a list of next indices
    #   a list of prob
    def step(self, ix):
        s = self.i2s(ix)
        b = self.pde.drift(s)
        ix_list = list(ix)
        
        ix_next = []; pr_next= []
        if self.is_interior(ix) and self.method=='cfd':
            for i in range(self.n_dim_):
                ix1 = ix_list; ix1[i]+=1; ix_next += [tuple(ix1),]
                pr1 = (1+2.*self.h_*b[i])/self.n_dim_/2.0; pr_next += [pr1,]
            for i in range(self.n_dim_):
                ix1 = ix_list; ix1[i]-=1; ix_next += [tuple(ix1),]
                pr1 = (1-2.*self.h_*b[i])/self.n_dim_/2.0; pr_next += [pr1,]
        
        return ix_next, pr_next
    
    ########running and terminal costs
    def run_h(self, ix):
        return self.pde.run(self.i2s(ix))*self.h_**2/self.n_dim_
    def term_h(self, ix):
        return self.pde.term(self.i2s(ix))
    
    
    ####Bellman equation and total loss
    #v is a function with torch tensor as input
    def bellman(self, ix, v):
        s = self.i2s(ix)
        lhs = v(torch.FloatTensor(s)); rhs = 0.
        #ipdb.set_trace()
        if self.is_interior(ix):
            rhs +=self.run_h(ix)
            ix_next, pr_next = self.step(ix)
            for ix1, pr1 in zip(ix_next, pr_next):
                rhs += pr1*v(torch.FloatTensor(self.i2s(ix1)))
            rhs *= self.discount_rate
        else:
            rhs = self.term_h(ix)
        return (rhs - lhs)

def solver(mdp, n_epoch = 500):
    ######### nn for value
    # Linear regression model
    value = nn.Sequential(
        nn.Linear(mdp.n_dim_, mdp.n_dim_+2),
        nn.ReLU(),
        nn.Linear(mdp.n_dim_+2, 2),
        nn.ReLU(),
        nn.Linear(2,1)
    )   
    print(value)
    
    # optimizer
    optimizer = torch.optim.SGD(value.parameters(), lr=0.01, momentum = .8) 
    
    #loss
    def tot_loss():
        out = 0.
        for ix in deep_iter(*mdp.v_shape_):
            out += mdp.bellman(ix,value)**2
        return out/mdp.v_size_
    
    print_n = 10
    epoch_per_print= int(n_epoch/print_n)
    
    start_time = time.time()
    for epoch in range(n_epoch):
        loss = tot_loss() #forward pass
        #backward propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % epoch_per_print == 0:
          print('Epoch [{}/{}], Loss: {:.4f}'.format(
                  epoch+1, n_epoch, loss.item()))
    end_time = time.time()
    print('>>>time elapsed is: ' + str(end_time - start_time))
    return value


#####test
if __name__=="__main__":
        
    p = Pde(n_dim_=1); m = Mdp(p, n_mesh_=16)
    value = solver(m, n_epoch=300)
    ######check solution
    err =0
    for ix1 in deep_iter(*m.v_shape_):
        s1 = m.i2s(ix1)
        v1 = value(torch.FloatTensor(s1)).item()
        err1 = v1-p.exact_soln(s1)
        err += err1**2
        #print(ix1, i2s(ix1), v1, exact_soln(s1),err1)
    
    err = err/m.v_size_
    print('>>>L2-error-norm: '+str(err))












