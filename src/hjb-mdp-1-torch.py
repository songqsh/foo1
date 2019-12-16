#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:10:06 2019

@author: songqsh
pytorch version

"""

import torch
import time
#import ipdb

#alternative for nditer, but for both tensor and ndarray
def deep_iter(data, ix=tuple()):
    try:
        for i, element in enumerate(data):
            yield from deep_iter(element, ix + (i,))
    except:
        yield ix, data



class Mdp:
    def __init__(
            self,
            dim=1,
            mesh_n=10,  # better to be even number
            lam=0.0,
            verbose=True
    ):
        self.dim = dim
        self.mesh_n = mesh_n
        self.h = 2.0 / mesh_n
        self.rate = dim / (dim + self.h ** 2 * lam)
        self.value = 0
        v_shape = (torch.ones(self.dim) * (self.mesh_n + 1)).type(torch.int32)
        self.value = torch.zeros(v_shape, dtype=torch.float64)  # initial
        self.a_scale = 2
        a_shape = (torch.ones(self.dim) * (self.mesh_n * self.a_scale + 1)).type(torch.int32)
        self.action = torch.zeros(a_shape, dtype=torch.float64)  # actions

        if verbose:
            print(str(dim) + '-d MDP from HJB')


    # convert index to state
    def i2s(self, tup):
        return (torch.tensor(tup, dtype=torch.float32) * self.h - 1.0).reshape(self.dim, 1)

        # convert index to action

    def i2a(self, tup):
        return (torch.tensor(tup, dtype=torch.float32) * self.h - self.a_scale).reshape(self.dim, 1)

    # define absorbing states
    # input: d-array for a state
    # return: true/false
    def is_absorbing(self, state):
        if state.shape != (self.dim, 1):
            print('warning: state dimension is not right')
            return False
        elif torch.max(torch.abs(state)) < 1:
            return False
        else:
            return True

    # value iteration
    # return
    #   state-value with boundary value
    def value_init(self):
        v0 = self.value
        for ix, elem in deep_iter(v0):
            s = self.i2s(ix)
            if self.is_absorbing(s):
                v0[ix] = -torch.sum(s**2)
    
    # define one step move
    # input:
    #   d-tuple for state
    #   d-tuple for action
    # return:
    #   2d-list for possible new state indices after one step
    #   2d-list for their corresponding probabilities
    #   float for instant reward

    def step(self, s_ind, a_ind):
        s0 = self.i2s(s_ind)
        a0 = self.i2a(a_ind)

        def ell(x,a):
            return self.dim + 2 * torch.sum(x ** 2) + torch.sum(a ** 2) * 0.5
        reward = self.h ** 2 * ell(s0, a0) / self.dim

        s1_ind = []

        if self.is_absorbing(s0):
            s1_ind.append(s_ind)
            pr.append(1.0)
        else:
            for i in range(self.dim):
                s1_ind_ = torch.tensor(s_ind, dtype=torch.int32)
                s1_ind_[i] += 1
                #s1_ind_.astype(int)
                s1_ind_ = tuple(s1_ind_.tolist())
                s1_ind.append(s1_ind_)
            for i in range(self.dim):
                s1_ind_ = torch.tensor(s_ind, dtype=torch.int32)
                s1_ind_[i] -= 1
                #s1_ind_.astype(int)
                s1_ind_ = tuple(s1_ind_.tolist())
                s1_ind.append(s1_ind_)
            pr = torch.tensor([1 + self.h * a0, 1 - self.h * a0], dtype=torch.float32)/(2*self.dim)
            pr = pr.tolist()

        return s1_ind, pr, reward
    
    # value iteration
    def value_iter(self):
        self.value_init()
        v0 = self.value
        v1 = v0.clone()

        iter_n = 1
        while True:
            for ix_s0, val in deep_iter(v0):
                s0 = self.i2s(ix_s0)
                if not self.is_absorbing(s0):
                    q1 = []
                    for ix_a, elem in deep_iter(self.action):
                        ix_s1, pr, rwd = self.step(ix_s0, ix_a)
                        rhs = rwd.item()
                        for k in range(2*self.dim):
                            #ipdb.set_trace()
                            rhs += v0[ix_s1[k]].item()*pr[k]
                        q1 += [rhs,]
                    v1[ix_s0] = self.rate*min(q1)
                    

            if torch.sum((v0 - v1) ** 2) < 1e-3:
                v0 = v1.clone()
                break
            v0 = v1.clone()
            iter_n += 1
        self.value = v0
        return iter_n    
    
    


if __name__ == "__main__":
    start_time = time.time()
    m = Mdp(mesh_n=10, dim=1)
    print('>>>number of iterations is: ' + str(m.value_iter()))
    end_time = time.time()
    print('>>>time elapsed is: ' + str(end_time - start_time))

    v = m.value
    print(v)

    err = 0.
    for ix, val in deep_iter(v):
        #ipdb.set_trace()
        s = m.i2s(ix)
        err1 = torch.abs(val + torch.sum(s**2))
        if err1>err:
            err = err1
    print('>>>sup norm is: ' + str(err))
