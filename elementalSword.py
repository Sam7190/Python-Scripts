# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:01:00 2020

@author: samir
"""
import numpy as np
import pandas as pd
from math import gcd

cities = ['Anafola', 'Benfriege', 'Demetry', 'Enfeir', 'Fodker', 'Glaser', 'Kubani', 'Pafiz', 'Scetcher', 'Starfex', 'Tamarania', 'Tamariza', 'Tutalu', 'Zinzibar']
connectivity = np.array([[ 0, 7, 3, 5, 7, 3, 5, 7, 5, 2, 2, 4, 6, 4],
                         [ 0, 0, 6,10, 7,10, 7, 5, 5, 9, 8, 3, 4,11],
                         [ 0, 0, 0, 4, 4, 6, 2, 4, 2, 4, 2, 3, 5, 6],
                         [ 0, 0, 0, 0, 7, 7, 4, 8, 6, 5, 2, 7, 9, 6],
                         [ 0, 0, 0, 0, 0,10, 3, 3, 2, 8, 6, 4, 5,10],
                         [12, 0, 0, 0, 0, 0, 8,10, 9, 5, 5, 7, 9, 6],
                         [ 0, 0,10, 0, 0, 0, 0, 5, 3, 6, 3, 4, 6, 7],
                         [ 0, 0, 0, 0, 8, 0, 0, 0, 2, 8, 6, 3, 2,10],
                         [ 0, 0, 6, 0, 6, 0,10, 7, 0, 6, 4, 2, 4, 8],
                         [ 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 8, 2],
                         [ 5, 0, 3, 6, 0, 0, 6, 0, 0, 9, 0, 5, 7, 4],
                         [10,10, 8, 0, 0, 0, 0, 3, 6, 0, 0, 0, 2, 8],
                         [ 0,12, 0, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0,10],
                         [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,10, 0, 0, 0]])
income = np.array([5, 1, 6, 2, 3, 1, 4, 3, 5, 3, 5, 5, 3, 1])

def proximitySort(connectivity, cities, city):
    if type(city) is int: city = cities[city]
    index = cities.index(city)
    vals = np.concatenate((connectivity[:index,index], connectivity[index, (index+1):]))
    C = np.delete(cities, index)
    args = np.argsort(vals)[::-1]
    print(vals[args])
    print(C[args])

def cityWeights(connectivity):
    return np.array([np.sum(connectivity[:i, i]) + np.sum(connectivity[i, (i+1):]) for i in range(len(connectivity))])

def citySkirmishes():
    return [np.unique(np.concatenate((connectivity.T[:i, i], connectivity.T[i, (i+1):])))[1:] for i in range(len(connectivity))]

def skirmPeeps():
    skirms = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, i:]))>0 for i in range(len(connectivity))]
    return skirms

def nonZero():
    P = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, (i+1):])) for i in range(len(connectivity))]
    return [P[i][P[i]>0] for i in range(len(P))]

def conn2dict():
    P = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, i:])) for i in range(len(connectivity))]
    d = {c: {} for c in cities}
    for i in range(len(P)):
        for j in range(len(P[i])):
            if P[i][j] > 0: d[cities[i]][cities[j]] = P[i][j]
    return d

Skirmishes = [None, {}]
def conn2set():
    P = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, i:])) for i in range(len(connectivity))]
    S = {}
    for i in range(len(P)):
        for j in range(len(P[i])):
            if P[i][j] > 0:
                S[frozenset([cities[i], cities[j]])] = P[i][j]
    return S
def getSkirmish():
    for S in Skirmishes[0]:
        if S in Skirmishes[1]:
            continue
        if np.random.rand() < (1 / (Skirmishes[0][S] + 1)):
            Skirmishes[1][S] = 3
    popS = []
    for S in Skirmishes[1]:
        Skirmishes[1][S] -= 1
        if Skirmishes[1][S] <= 0:
            popS.append(S)
    for S in popS:
        Skirmishes[1].pop(S)
def notAtWar():
    s = set()
    for S in Skirmishes[1]:
        for city in S:
            s.add(city)
    return set(cities).difference(s)

def citySkirmProbs():
    skirms = nonZero()
    atone, i = [], 0
    for a in skirms:
        P = 1/(a+1)
        p = 1
        for Pr in P:
            p *= (1 - Pr)
        atone.append((1-p))
        i += 1
    d = pd.DataFrame(cities)
    d[1] = atone
    d[2] = income
    d[3] = [(1-atone[i])*income[i] for i in range(len(atone))]
    d[4] = skirms
    return d

def lcm(array):
    lcm = array[0]
    for i in array[1:]:
      lcm = lcm*i//gcd(lcm, i)
    return lcm

def findAverageIncome(skirmishTurns, incomePerTurn):
    skirmishTurns = np.array(skirmishTurns) + 2
    G = lcm(skirmishTurns)
    lostIncome = np.sum(G / skirmishTurns)*2
    return (G*incomePerTurn - lostIncome) / G

def averageIncomes(connectivity, income):
    skirmishes = citySkirmishes(connectivity)
    print([findAverageIncome(skirmishes[i], income[i]) for i in range(len(income))])
    
def requiredStages(rows, maximum=120):
    a = np.ones((rows,),dtype=int)
    total = np.sum(a)
    while total < maximum:
        a += 1
        total += np.sum(a)
    return a[0], total
    
        