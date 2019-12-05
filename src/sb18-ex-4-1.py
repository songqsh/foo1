#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:36:40 2019

@author: songqsh
"""

import numpy as np

class gridworld:
    def __init__(
            self,
            WORLD_SIZE = 4, #length of each side
            DIM = 2, #dimension number
            verbose = True
            ):
        self.WORLD_SIZE = WORLD_SIZE
        self.DIM = DIM
        self.ACTIONS = np.append(np.eye(DIM), -np.eye(DIM), axis=0) #each row is one action
        self.ACTION_PROB = 1./(2*DIM) #random policy 
        if verbose:
            print(str(DIM) + 
                  '-d Gridworld, \n length of each side: '
                  + str(WORLD_SIZE)
                  + '\n reflecting boundary \n and absorbing corner'
                  )
        
    #state: n-d array 
    #return: true (if reflecting) or false.
    def is_reflecting(self, state): 
      out = 0 #false by dafault
      out = out or np.any(state>self.WORLD_SIZE-1)
      out = out or np.any(state < 0)
      return  out


    #state: n-d array 
    #return: true (if absorbing/terminating) or false.
    def is_absorbing(self, state): 
      out = 0
      out = out or np.all(state == 0)
      out = out or np.all(state == self.WORLD_SIZE-1)
      return  out
      
    
    #input
    #state: n-d np.array
    #action: n-d np.array
    #return:
    #new_state: n-d np.array, 
    #           if it is terminal, then stay
    #           if next move is absorbing, then stay in previous state
    #           otherwise state + action
    #reward: -1 for each move
    
    def step(self, state, action):
      next_state = state+action
      if self.is_absorbing(state) or self.is_reflecting(next_state):
        next_state = state
      reward = -1
      return next_state, reward
        
        
    #value iteration
    #return:
      #v0: state value matrix
      #iter_n: number of iterations.
    def value_iteration(self):
      v_shape = (np.ones(self.DIM)*self.WORLD_SIZE).astype(int)
      v0 = np.zeros(shape=v_shape)
      v1 = v0.copy()
    
      iter_n = 1
      while True:
        it = np.nditer(v0, flags=['multi_index'])
        while not it.finished:
          state0 = np.array(it.multi_index)
          if self.is_absorbing(state0):
            v1[it.multi_index] = 0.
          else:
            rhs = 0
            for a in self.ACTIONS:
                state1, reward = self.step(state0, a)
                state1_tuple = tuple([i for i in state1.astype(int)])
                rhs += self.ACTION_PROB*(reward+v0[state1_tuple])
            v1[it.multi_index]= rhs 
          it.iternext()
          
        if np.sum(np.abs(v1-v0)) < 1e-4:
          v0 = v1.copy()
          break
        v0 = v1.copy()
        iter_n += 1
    
      return v0, iter_n
        

    
    
            

        


