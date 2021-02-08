# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:01:00 2020

@author: samir
"""
import numpy as np
import pandas as pd
from math import gcd
import matplotlib.pyplot as plt
from matplotlib import rc
import addcopyfighandler
from matplotlib.patches import Circle


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
    
nodes = {'fodker': [(2, 1)],'kubani': [(7, 1)],'scetcher': [(4, 4)],'pafiz': [(1, 6)],'tamariza': [(4, 7)],'zinzibar': [(15, 2)],'enfeir': [(13, 1)],
             'tamarania': [(11, 3)],'demetry': [(8, 4)],'starfex': [(15, 6)],'anafola': [(11, 7)],'tutalu': [(2, 10)],'benfriege': [(5, 13)],'glaser': [(13, 13)]}

edges = {'anafola-glaser':[13], 'anafola-starfex':[7], 'anafola-tamarania':[6], 'anafola-tamariza':[11],
         'benfriege-tamariza':[11], 'benfriege-tutalu':[13],
         'demetry-kubani':[11], 'demetry-scetcher':[7], 'demetry-tamarania':[4], 'demetry-tamariza':[9],
         'enfeir-tamarania':[7], 'enfeir-zinzibar':[2],
         'fodker-pafiz':[9], 'fodker-scetcher':[7],
         'kubani-scetcher':[11], 'kubani-tamarania':[7],
         'pafiz-scetcher':[8], 'pafiz-tamariza':[4], 'pafiz-tutalu':[8],
         'scetcher-tamariza':[7], 
         'starfex-tamarania':[10],
         'tamarania-zinzibar':[11],
         'tamariza-tutalu':[7]}

xpix, ypix = 1, 396/343
xsize, ysize = 1, 1
scale = 1
xprel, yprel = scale * (xpix / xsize), scale * (ypix / ysize)

def get_dim(dx, dy):
    return int(xpix*dx), int((ypix * (dy-(dy//2))) + (xpix * (dy//2) / np.sqrt(3)))

def get_pos(x, y):
    if y % 2:
        xoffset, yoffset = xpix/2, (xpix/np.sqrt(3)) + (ypix/2 - xpix/(2*np.sqrt(3)))
    else:
        xoffset, yoffset = 0, 0
    return (x * xpix) + xoffset, (y // 2) * (ypix + xpix/np.sqrt(3)) + yoffset

def get_posint(x, y):
    px, py = get_pos(x, y)
    return int(px), int(py)

def get_relpos(x, y):
    px, py = get_pos(x, y)
    return float(px / xsize), float(py / ysize)

city_labor = {'anafola':{'Persuasion':5, 'Excavating':5},
             'benfriege':{'Critical Thinking':5, 'Persuasion':5, 'Crafting':8, 'Survival':5},
             'demetry':{'Bartering':8, 'Crafting':5},
             'enfeir':{'Critical Thinking':5, 'Heating':5, 'Smithing':5, 'Stealth':8},
             'fodker':{'Bartering':5, 'Smithing':5},
             'glaser':{'Critical Thinking':5, 'Persuasion':5, 'Crafting':5, 'Survival':5, 'Excavating':5},
             'kubani':{'Critical Thinking':5, 'Bartering':5, 'Crafting':5, 'Gathering':5},
             'pafiz':{'Persuasion':8, 'Crafting':5, 'Heating':5, 'Gathering':5},
             'scetcher':{'Smithing':5, 'Stealth':5},
             'starfex':{'Heating':5, 'Gathering':5, 'Excavating':5},
             'tamarania':{'Smithing':8},
             'tamariza':{'Critical Thinking':5, 'Persuasion':5, 'Heating':5},
             'tutalu':{'Smtihing':5, 'Excavating':5},
             'zinzibar':{'Persuasion':5, 'Smithing':5, 'Survival':8}}
skills = ['Persuasion', 'Critical Thinking', 'Heating', 'Survival', 'Smithing', 'Crafting', 'Excavating', 'Stealth', 'Gathering', 'Bartering']
skill_users = {'Persuasion':'Politician', 'Critical Thinking':'Librarian', 'Heating':'Chef', 'Survival':'Explorer', 'Smithing':'Smith', 'Crafting':'Innovator', 'Excavating':'Miner', 'Stealth':'General', 'Gathering':'Huntsman', 'Bartering':'Merchant'}

def rbtwn(mn, mx):
    return np.random.choice(np.arange(mn, mx+1))

def generateSkirmish(radius=0.6, textsize=10, legend=True, token_radius=0.21, prob_textsize=10, show_prob=False):
    #rc('text', usetex=True)
    fig = plt.figure(figsize=(13.5,8.5))
    ax = fig.add_subplot(111)
    left, right, bottom, top = float('inf'), -float('inf'), float('inf'), -float('inf')
    # Plot Edges
    for edge, v in edges.items():
        city1, city2 = edge.split('-')
        x1, y1 = get_pos(*nodes[city1][0])
        x2, y2 = get_pos(*nodes[city2][0])
        if legend:
            ax.plot([x1, x2], [y1, y2], color='k', lw=3, zorder=0)
            x, y = (x1 + x2)/2, (y1 + y2)/2
            ax.add_patch(Circle((x, y), token_radius, zorder=1, fc='w', ec='w'))
            ax.text(x, y, f'1/{v[0]}', fontsize=prob_textsize, ha='center', va='center', zorder=2)
        else:
            r = np.random.rand()
            if r <= (1 / (v[0]+2)):
                clr = 'k'
            elif r <= (1 / v[0]):
                clr = 'C0'
            else:
                clr = 'w'
            ax.plot([x1, x2], [y1, y2], color=clr, lw=3, zorder=0)
            if show_prob:
                x, y = (x1 + x2)/2, (y1 + y2)/2
                ax.add_patch(Circle((x, y), token_radius, zorder=1, fc='w', ec='w'))
                ax.text(x, y, f'{np.round(r, 2)} | {np.round(1/v[0],2)}', fontsize=prob_textsize, ha='center', va='center', zorder=2)
    # Plot Nodes
    for city, v in nodes.items():
        x, y = get_pos(*v[0])
        ax.add_patch(Circle((x, y), radius, fc='w', ec='k', zorder=1))
        left, right = np.min([left, (x-radius)]), np.max([right, (x+radius)])
        bottom, top = np.min([bottom, (y-radius)]), np.max([top, (y+radius)])
        ax.text(x, y, city[0].upper()+city[1:], fontsize=textsize, ha='center', va='center', zorder=2)
    xdiff, ydiff = (right - left)*0.005, (top - bottom)*0.005
    # List Jobs
    if not legend:
        jobList = []
        for city in cities:
            assistants = [r'$'+city+r'$:']
            for skill in skills:
                val = city_labor[city.lower()][skill] if skill in city_labor[city.lower()] else 2
                if rbtwn(1, 10) <= val:
                    assistants.append(skill_users[skill])
            if len(assistants) == 1: assistants.append('--')
            assistants.append('')
            if len(assistants) > 2: jobList.append(assistants)
        split = int(np.ceil(len(jobList)/2))
        col1 = [item for sublist in jobList[:split] for item in sublist]
        col2 = [item for sublist in jobList[split:] for item in sublist]
        ax.text(right + xdiff, (bottom + top)/2, '\n'.join(col1), ha='left', va='center', fontsize=textsize)
        ax.text(right + (right - left)*0.1, (bottom + top)/2, '\n'.join(col2), ha='left', va='center', fontsize=textsize)
    else:
        translation = sorted(skill_users.items(), key=lambda kv: kv[1])
        col = [t[1]+': '+t[0] for t in translation]
        col = [r'$Assistant: Skill$'] + col
        ax.text(right + xdiff, (bottom + top)/2, '\n\n'.join(col), ha='left', va='center', fontsize=textsize)
    ax.set_xlim(left - xdiff, right + (right - left)*0.2)
    ax.set_ylim(bottom - ydiff, top + ydiff)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show()