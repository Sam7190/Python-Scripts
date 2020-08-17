# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:33:37 2020

@author: samir
"""
import numpy as np

# Helper Functions
def rbtwn(mn, mx, amt=1):
    size = None if amt <= 1 else amt
    return np.random.choice(np.arange(mn, mx+1), size)

def isbetween(mn, mx, numb):
    return (numb >= mn) * (numb <= mx)

def fset(tpl):
    return frozenset(np.arange(tpl[0], tpl[1]+1))

def clearGrid(Player):
    Player.parentBoard.game_page.clear_actionGrid()
def restoreGrid(Player):
    Player.parentBoard.game_page.restore_actionGrid()
    
def give_item(Player, item):
    def update_item():
        Player.add_item(item)
    return update_item

def train(Player, atr_or_skl, trainer_type):
    # Trainer types: Adept, City Master, Outskirt Master
    def train_player():
        Player.train(atr_or_skl, trainer_type)
    return train_player


        
def gather(Player, item):
    

# Define Action Functions
# 1. Road: No Action

# 2. Plains
def E_plains(Player):
    clearGrid(Player)
    
    result = {fset((1,1)):train(Player, {'Agility','Gathering'}, 'Outskirt Master'),
              fset((2,6)):}
    