# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 16:32:30 2020

@author: samir
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as LG
from sklearn.ensemble import RandomForestClassifier as RF

verbose = False
model = 'RF'
training_cols = 11
trainprc = 1
realOpacity = 0.96


def trainLoc(length, prc=trainprc):
    z = np.zeros((length,),dtype=bool)
    I = np.random.choice(np.arange(length), int(length*prc))
    z[I] = True

def loadData(filename='data\\DefenseLog.csv'):
    D = pd.read_csv(filename)
    T = D[D.columns[:training_cols]]
    return T, D['clicked'], D['dodged']

def trainRF(T, c, d, onlyD=False, trees=30, min_samples_leaf=5):
    if not onlyD:
        crf = RF(trees, min_samples_leaf=min_samples_leaf)
        crf.fit(T, c)
    else:
        crf = None
    drf = RF(trees, min_samples_leaf=min_samples_leaf)
    drf.fit(T[c==1], d[c==1])
    return crf, drf

def trainLG(T, c, d, onlyD=False, max_iter=500):
    if not onlyD:
        clg = LG(solver='lbfgs', max_iter=max_iter)
        clg.fit(T, c)
    else:
        clg = None
    dlg = LG(solver='lbfgs', max_iter=max_iter)
    dlg.fit(T[c==1], d[c==1])
    return clg, dlg

def LOOCV(T, c, d, fn):
    real = T['opacity'] == realOpacity
    print("Human Real Click AUROC", roc_auc_score(real, c))
    print("Human Real Dodge AUROC", roc_auc_score(real[c==1], d[c==1]))
    cyp, dyp = np.zeros((len(T),)), np.zeros((len(T[c==1]),))
    loc = np.ones((len(T),),dtype=bool)
    for i in range(len(T)):
        print(f'\rClick LOOCV iteration {str(i).ljust(len(str(len(T))))}/{len(T)}',flush=True, end='')
        loc[i] = False
        loc[i-1] = True
        cf, df = fn(T[loc], c[loc], d[loc])
        cyp[i] = cf.predict_proba(T[~loc])[:,1][0]
    print(f'\rClick LOOCV iteration {len(T)}/{len(T)}',flush=True)
    loc = np.ones((len(T[c==1]),),dtype=bool)
    for i in range(len(T[c==1])):
        print(f'\rDodge LOOCV iteration {str(i).ljust(len(str(len(T[c==1]))))}/{len(T[c==1])}',flush=True, end='')
        loc[i] = False
        loc[i-1] = True
        cf, df = fn(T[c==1][loc], c[c==1][loc], d[c==1][loc], True)
        dyp[i] = df.predict_proba(T[c==1][~loc])[:,1][0]
    print(f'\rDodge LOOCV iteration {len(T[c==1])}/{len(T[c==1])}',flush=True)
    print("---- Box Click ----")
    print("AI Click AUROC", roc_auc_score(c, cyp), "AI Real Click AUROC", roc_auc_score(real, cyp))
    print("Human Avg Real Click", np.mean(c[real==1]), "AI Avg Real Click", np.mean(cyp[real==1]))
    print("Human Avg Fake Click", np.mean(c[real==0]), "AI Avg Fake Click", np.mean(cyp[real==0]))
    print("---- Dodge ----")
    print("AI Dodge AUROC", roc_auc_score(d[c==1], dyp), "AI Real Dodge AUROC", roc_auc_score(real[c==1], dyp))
    print("Human Avg Real Dodge", np.mean(d[(c==1)&(real==1)]), "AI Avg Real Dodge", np.mean(dyp[real[c==1]==1]))
    print("Human Avg Fake Dodge", np.mean(d[(c==1)&(real==0)]), "AI Avg Fake Dodge", np.mean(dyp[real[c==1]==0]))

models = {'RF': trainRF,
          'LG': trainLG} 

T, c, d = loadData()
cmdl, dmdl = models[model](T, c, d)

if verbose:
    LOOCV(T, c, d, models[model])
