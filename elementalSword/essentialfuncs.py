# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:45:09 2024

@author: samir
"""
import numpy

def get_hexcolor(rgb):
    return '%02x%02x%02x' % rgb
def randofsum(s, n):
    return np.random.multinomial(s,np.ones(n)/n,size=1)[0]
def euc(a, b): 
    return np.linalg.norm(a-b)
def isint(s):
    try:
        return int(s) == float(s)
    except ValueError:
        return False