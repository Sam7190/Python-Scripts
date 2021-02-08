# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:31:21 2020

@author: samir.farooq
"""
import numpy as np
import pyperclip

lbls = ['Attack', 'Hit Points', 'Agility', 'Cunning', 'Technique', 'Stability', 'Def - Physical', 'Def - Elemental', 'Def - Wizard', 'Def - Trooper']
maxs = {'Agility':15, 'Cunning':15, 'Technique':12}

def randofsum(s, n):
    return np.random.multinomial(s,np.ones(n)/n,size=1)[0]

def npc_stats(lvl, fixed=None, maxAtr=None):
    conditions = [None, None, None, None]
    n = len(lbls)
    if ('Hit Points' in lbls):
        hi = lbls.index('Hit Points')
        conditions[0] = f'a[{hi}]==0'
    if ('Attack' in lbls) and ('Stability' in lbls):
        ai, si = lbls.index('Attack'), lbls.index('Stability')
        conditions[1] = f'a[{si}]>a[{ai}]'
    if ('Attack' in lbls):
        ai = lbls.index('Attack')
        conditions[2] = f'a[{ai}]==0'
    if ('Technique' in lbls):
        ti = lbls.index('Technique')
        conditions[3] = f'a[{ti}]>12'
    if fixed is not None:
        for atr, fixedlvl in fixed.items():
            conditions.append(f'a[{lbls.index(atr)}]!={fixedlvl}')
    if maxAtr is not None:
        for atr, maxLvl in maxAtr.items():
            conditions.append(f'a[{lbls.index(atr)}]>{maxLvl}')
    while True:
        a = randofsum(lvl, n)
        breakout = True
        for i in range(len(conditions)):
            if (conditions[i] is not None) and eval(conditions[i]):
                breakout = False
                break
        if breakout: break
    return a

def generate(groups, minLvl, lvlRange, rows=3, fixed=None, maxAtr=maxs, uniqueGroups=None):
    header = ','.join(['Group', 'Range', 'Level'] + lbls)
    D, columns = [], []
    if uniqueGroups is None:
        uniqueGroups = groups
    for g in range(groups):
        if not (g % rows):
            columns.append([header])
            col = (g + 1)//rows
        first = True
        gMinLvl = minLvl + lvlRange * (g % uniqueGroups)
        for i in range(lvlRange):
            lvl = gMinLvl + i
            stats = npc_stats(lvl, fixed=fixed, maxAtr=maxAtr)
            if first:
                group = g + 1
                first = False
            else:
                group = ''
            columns[col].append(','.join([str(k) for k in np.concatenate(([group], [i+1], [lvl], stats))]))
    for i in range(len(columns[0])):
        D.append(','.join([columns[j][i] for j in range(col+1)]))
    pyperclip.copy('\n'.join(D))
    
def arena(groups, lvlRange=11, rows=3, maxAtr=maxs):
    Lb = []
    for l in lbls:
        if l in maxAtr:
            Lb.append(l + f' (max={maxAtr[l]})')
        else:
            Lb.append(l)
    header = ','.join(['Group', 'Range', 'Level Difference'] + Lb)
    D, columns = [], []
    for g in range(groups):
        if not (g % rows):
            columns.append([header])
            col = (g + 1)//rows
        first = True
        for i in range(lvlRange):
            lvl = 100 + (i - (lvlRange//2))
            stats = npc_stats(lvl)
            stats2 = [f'+{l-10}' if (l-10)>=0 else f'{l-10}' for l in stats]
            if first:
                group = g + 1
                first = False
            else:
                group = ''
            columns[col].append(','.join([k for k in np.concatenate(([group], [i+1], [(i-(lvlRange//2))], stats2))]))
    for i in range(len(columns[0])):
        D.append(','.join([columns[j][i] for j in range(col+1)]))
    pyperclip.copy('\n'.join(D))