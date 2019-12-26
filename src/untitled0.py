#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:51:20 2019

@author: songqsh
"""

# Python program to demonstrate working 
# of map. 

# Return double of n 
def addition(n): 
	return n + n 

# We double all numbers using map() 
numbers = (1, 2, 3, 4) 
result = map(addition, numbers) 
#print(list(result)) 

####paras for pde
n_dim_ = 2
lam_ = 0.

def run(s):
    #return sum([a**2 for a in s])*(lam_- n_dim_)
    return sum(map(lambda a: a**2, s))