# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

random walk within a given shape of tuple
"""

import random

class grid:
    def __init__(self, shape = (5, 5)):
        self.n_dim = len(shape)
        self.shape = shape
        print('>>> grid world shape is: '+str(shape))

    def is_interior(self,ix):
        return all([0.<a<b for a,b in zip(ix1,list(self.shape))])
     
    #input: lists of index
    #return: running cost, list of next index, list of probability
    def step(self, ix):
        run_cost = 0.                
        ix_next = []; pr_next= []
        if self.is_interior(ix):
            run_cost = 1.
            for i in range(self.n_dim):
                ix1 = ix.copy(); ix1[i]+=1; ix_next += [ix1,]
                pr1 = 1./(self.n_dim*2.0) 
                pr_next += [pr1,]
            for i in range(self.n_dim):
                ix1 = ix.copy(); ix1[i]-=1; ix_next += [ix1,]
                pr1 = 1./(self.n_dim*2.0) 
                pr_next += [pr1,]
     
        return run_cost, ix_next, pr_next
    
    def step_random(self, ix):
        run_cost, ix_next, pr_next = self.step(ix)
        ix_next_rd = random.choices(ix_next, pr_next, k = 1)
        return run_cost, ix_next_rd[0]
        


#####check
g1 = grid(shape=(5,5))
ix1 = [2,3]
print(g1.is_interior(ix1))
o1, o2, o3 = g1.step(ix1)
print(o1, '\n', o2, '\n', o3)
tot_cost = 0.


while g1.is_interior(ix1):
    run_cost, ix1 = g1.step_random(ix1)
    tot_cost+=run_cost
    
print(tot_cost)
    