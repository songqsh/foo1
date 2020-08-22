#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:56:17 2020

@author: qsong
"""


# option pricing with fft

import numpy as np
#import matplotlib.pyplot as plt
from scipy.integrate import quad

class BSM:
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        
    #characteristic function for $ln S_T$
    def charfun(self, T, u):
        o1 = np.log(self.S0) + (self.r - self.sigma**2/2.0)*T
        o2 = self.sigma**2*T/2.
        o3 = 1j*u*o1 - u**2*o2
        return np.exp(o3)
    
    
    
# Heston model    
class Heston:
    def __init__(self, S0, v0, r, kappa, theta, sigma, rho):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
    #char func for ln(S_T) from Ng05
    def charfun(self, T, u):
        t1 = self.kappa - self.rho*self.sigma*1j*u
        D = np.sqrt(t1**2+(u**2+1j*u)*self.sigma**2)
        G = (t1-D)/(t1+D)
        t2 = 1-G*np.exp(-D*T)
        f1 = np.exp(1j*u*(np.log(self.S0+self.r*T)))
        f2 = np.exp(self.v0*(1-np.exp(-D*T))*(t1-D)/self.sigma**2/t2)
        f3 = np.exp(self.kappa*self.theta*(
            T*(t1-D)-2*np.log(t2/(1-G)))/self.sigma**2)
        return f1*f2*f3
   

class FTCall:
    def __init__(self, model):
        self.md = model
        
    #$\psi$ function for carr-madan method
    def psi(self, w, T, alpha):
        o1 = np.exp(-self.md.r*T)
        o1 = o1*self.md.charfun(T, w - (alpha+1)*1j)
        o2 = alpha**2+alpha-w**2+1j*(2*alpha+1.)*w
        return o1/o2
    
    #carr-madan method with damping
    def price_cm(self, K, T, alpha = 1.5):
        k = np.log(K)
        integrand = lambda w: (np.exp(-1j*w*k)*self.psi(w, T, alpha)).real
        integral = quad(integrand, 0, np.inf)[0]
        return np.exp(-alpha*k)/np.pi*integral
    
# =============================================================================
# 
# #### bsm test    
bsm = BSM(100, 0.0475, 0.2)
ftc = FTCall(bsm)
ans = ftc.price_cm(110, 1., alpha = 1.5)
print(
      f'BSM FT price by carr-madan is \n >> {ans}'
      )
# =============================================================================

######### Heston test
# =============================================================================
hes = Heston(100, 0.0175, 0., 1.5768, 0.0398, 0.5751, -0.5751)
ftc = FTCall(hes)
ans = ftc.price_cm(80, 10, alpha = 1.5)
print(
      f'Heston FT price by carr-madan is \n  >> {ans}'
      )
print('(ref. P61 of [Hirsa13])')
# =============================================================================




