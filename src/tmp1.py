#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:56:36 2019

@author: songqsh
"""

class animal:
    def __init__(
            self,
            color = 'red',
            kind = 'dog',
            ):
        self.color = color
        self.kind = kind
            
        def sound(repeat = 2):
            for i in range(repeat):
                if kind=='dog':
                    print('woof')
                