#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:52:47 2020

@author: songqsh
"""
from hjb_mdp_v01 import *

#### test Pde
print('test begins')
p = Pde()
s =[.5, .3]; a = [.2, 0.8]
print(p.drift(s,a), p.run_cost(s,a), p.term_cost(s))
print('test ends')
####test end


    
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

#### test 191222epde.pdf
n_dim = 2; lam = 0
def fun_b(x):
    return 0
def fun_ell(x):
    return float(-n_dim)

m1 = Mdp(n_mesh = 8)
#def drift(self,s,a):
#    return [0]*self.n_dim
m1.drift = lambda self, s, a: [0]*self.n_dim
print(m1.drift(s,a))

    