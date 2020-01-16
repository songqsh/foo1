#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:34:12 2020

@author: songqsh
"""

import scipy.optimize as so

def fun(c, x):
    return sum((x1+c1)**2 for x1, c1 in zip(x,c))

print(so.fmin(fun, c, args = (c,), disp = 0))