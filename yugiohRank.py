# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:47:46 2020

@author: samir.farooq
"""
import numpy as np

def avgRank(Ls):
    d, D = {}, {}
    for L in Ls:
        for i in range(len(L)):
            l = L[i]
            if l in d:
                d[l].append(i)
            else:
                d[l] = [i]
    for l in d:
        D[l] = np.mean(d[l])
    print(sorted(D.items(), key=lambda item: item[1]))