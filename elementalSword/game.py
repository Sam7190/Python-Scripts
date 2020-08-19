"""
cd Documents\\GitHub\\Python-Scripts\\elementalSword
python game.py

"""

import numpy as np
from PIL import Image as pilImage
import os
import sys
from functools import partial # This would have made a lot of nested functions unnecessary! (if I had known about it earlier)
from copy import deepcopy
import socket_client
from time import time
#import mechanic as mcnc
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
# to use buttons:
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color,Rectangle,Ellipse
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.core.window import Window

# Tile Resolution and number of tiles in gameboard.
seed = 303
xpix = 343
ypix = 396
xtiles = 17
ytiles = 15

def lclPlayer():
    return game_app.game_page.board_page.localPlayer
def rbtwn(mn, mx, size=None):
    return np.random.choice(np.arange(mn, mx+1), size)

def encounter(name, lvlrange, styles, reward, fixed=None, party_size=1, consume=None, action_amt=1, empty_tile=False):
    P = lclPlayer()
    # Pause all buttons so only fighting screen remains
    P.paused = True
    # [INCOMPLETE] Conduct Fight
    lvl = rbtwn(lvlrange[0],lvlrange[1])
    cbstyle = styles[rbtwn(0,len(styles)-1)]
    output(f'Encounter Lv{lvl} {name} using {cbstyle}','yellow')
    # Allow buttons to be pressed again
    P.paused = False
    # For now exit the action loop - but we will want this to be different depending on circumstances
    exitActionLoop(consume, action_amt, empty_tile)()

def isbetween(mn, mx, numb):
    return (numb >= mn) * (numb <= mx)

def output(message, color=None):
    game_app.game_page.update_output(message, color)
    print(message)
def actionGrid(funcDict, save_rest):
    game_app.game_page.make_actionGrid(funcDict, save_rest)
def check_paralysis():
    P = lclPlayer()
    if (P.fatigue > 10) or (P.paralyzed_rounds in {1,2}):
        P.paralyzed_rounds += 1
        P.actions = 0
        for a in range(P.max_actions):
            P.recover(None)
    elif P.paralyzed_rounds > 2:
        P.paralyzed_rounds = 0
def exitActionLoop(consume=None, amt=1, empty_tile=False):
    P = lclPlayer()
    def exit_loop(_=None):
        if P is not None:
            if consume == 'road':
                P.road_moves -= amt
                if P.road_moves == 0:
                    P.takeAction(amt)
                    P.road_moves = P.max_road_moves
            elif (consume == 'buy') or (consume == 'bartered'):
                P.avail_transactions -= amt
                if consume == 'bartered': P.activated_bartering=True
                if P.avail_transactions == 0:
                    # If the player makes his max transactions, then it is the same as if he bartered. Otherwise, bartering is handled in the takeAction method.
                    P.activated_bartering = False
                    P.takeAction(amt)
                    P.avail_transactions = P.max_avail_transactions
#                else:
#                    P.update_mainStatPage()
            elif amt>0:
                P.takeAction(amt)
            if empty_tile: P.currenttile.empty_tile()
            if (consume is None) or ('tiered' not in consume):
                P.go2action()
            else:
                # Otherwise the player must be descended into a cave or ontop of a mountain
                tier = int(consume[-1])
                P.tiered = True if tier > 1 else False
                P.go2action(tier)
    return exit_loop

def getItem(item, amt=1, consume=None, empty_tile=False, action_amt=1):
    P = lclPlayer()
    def getitem(_=None):
        if P.paused:
            return
        P.addItem(item, amt)
        exitActionLoop(consume, action_amt, empty_tile)()
    return getitem

def sellItem(item, to_trader, barter=None, _=None):
    P = lclPlayer()
    if P.paused:
        return
    print(item)
    for categ in gameItems:
        if item in gameItems[categ]:
            sellprice = sellPrice[gameItems[categ][item]]
            break
    if barter is None:
        output("Would you like to barter?",'blue')
        actionGrid({'Yes':partial(sellItem, item, to_trader, True), 'No':partial(sellItem, item, to_trader, False)}, False)
    elif barter == True:
        bartering = P.activateSkill("Bartering")
        r = rbtwn(1,12)
        if r <= bartering:
            P.useSkill("Bartering")
            output("You successfully Barter", 'green')
            if (bartering > 8) and to_trader:
                P.coins += (sellprice+2)
            else:
                P.coins += (sellprice+1)
            # Its basically inverted getItem
            getItem(item, -1, 'bartered', False, 1)()
        else:
            output("You failed to barter, sell anyway?", 'red')
            actionGrid({'Yes':partial(sellItem, item, to_trader, False), 'No':exitActionLoop('bartered', 0, False)}, False)
    else:
        P.coins += sellprice
        getItem(item, -1, 'buy', False)()

def buyItem(item, cost, from_trader, amt=1, consume='buy'):
    P = lclPlayer()
    def buyitem(_=None):
        if P.paused:
            return
        if P.coins < (cost*amt):
            output("Insufficient funds!",'yellow')
        else:
            P.coins -= (cost*amt)
            if from_trader and (item in P.currenttile.trader_wares):
                P.currenttile.buy_from_trader(item)
            getItem(item, amt, consume, False, 1)()
    return buyitem

def Trading(items, trader, barter=False):
    # [INCOMPLETE] Selling is not yet implemented
    P = lclPlayer()
    if P.paused:
        return
    def activate_bartering(_):
        if P.paused:
            return
        bartering = P.activateSkill('Bartering')
        r = rbtwn(1, 12)
        if r <= bartering:
            P.useSkill('Bartering')
            output("You successfully Barter", 'green')
            if (bartering > 8) and trader:
                Trading(items, trader, 2) # Reduce prices by max of 2
            else:
                Trading(items, trader, 1)
        else:
            Trading(items, trader, None) # Not given the opportunity to barter again unless performs a different action first
    actions = {}
    for item in items:
        for categ in gameItems:
            if item in gameItems[categ]:
                cost = gameItems[categ][item] if trader else mrktPrice[gameItems[categ][item]]
                cost = cost if (barter is None) or (not barter) else cost-barter
                clr = 'ffff75' if (barter is None) or (not barter) else '00ff75'
                actions[f'{item}:[color={clr}]{cost}[/color]'] = buyItem(item, cost, trader, consume='buy' if barter==False else 'bartered')
                # Give the player the option to go back to regular action menu of the tile
                actions['Back'] = exitActionLoop(None, 0)
    # If the player has not activated bartering already, give them a chance to do so
    if (barter == False) and (P.activated_bartering==False): actions['Barter'] = activate_bartering
    actionGrid(actions, False)
    
def await_trading(items, trader, barter=False):
    def trade(_):
        Trading(items, trader, barter)
    return trade

def Train(abilities, master, confirmed):
    P = lclPlayer()
    if P.paused:
        return
    if not confirmed:
        if master == 'monk':
            cost = 0
        elif type(abilities) is str:
            if master == 'adept':
                cost = P.get_level(abilities)+5
            elif master == 'city':
                cost = P.get_level(abilities)+11
            else:
                cost = 8 if P.get_level(abilities) < 8 else 12
        else:
            costs = []
            for ability in abilities:
                if master == 'adept':
                    cost = P.get_level(abilities)+5
                elif master == 'city':
                    cost = P.get_level(abilities)+11
                else:
                    cost = 8 if P.get_level(abilities) < 8 else 12
                costs.append(f'{cost} for {ability}')
            cost = '('+', '.join(costs)+')'
        output("Would you like to train at cost of {cost} coins?")
        game_app.game_page.make_actionGrid({'Yes':(lambda _:Train(abilities, master, True)), 'No':exitActionLoop()}, False)
    elif type(abilities) is not str:
        output("Which skill would you like to train?")
        game_app.game_page.make_actionGrid({ability:(lambda _:Train(ability, master, confirmed)) for ability in abilities})
    else:
        lvl = P.get_level(abilities)
        if master == 'adept':
            cost = lvl+5
            if lvl >= 8:
                output("Adept trainers cannot teach beyond lvl 8",'yellow')
                # Should take you back to tile actions without consuming an action or taking fatigue
                exitActionLoop(None,0)()
            elif P.coins < cost:
                output("You do not have sufficient funds.",'yellow')
                exitActionLoop(None,0)()
            elif rbtwn(1,10) <= P.fatigue:
                output("You were unable to keep up with training.",'red')
                P.takeAction()
                Train(abilities, master, False)
            elif rbtwn(1,3) == 1:
                output(f"You successfully leveled up in {abilities}",'green')
                P.updateSkill(abilities,1)
                exitActionLoop(None,1)()
            else:
                critthink = P.activateSkill('Critical Thinking')
                if rbtwn(1,12) <= critthink:
                    P.useSkill('Critical Thinking')
                    output(f"You successfully leveled up in {abilities}",'green')
                    P.updateSkill(abilities,1)
                    exitActionLoop(None,1)()
                else:
                    output("You were unsuccessful.",'red')
                    P.takeAction()
                    Train(abilities, master, False)
        else:
            if master == 'city':
                cost = lvl+11
            elif master == 'monk':
                cost = 0
            else:
                cost = 10
            if P.coins < cost:
                output("You do not have sufficient funds.",'yellow')
                exitActionLoop(None,0)()
            elif rbtwn(2 if master=='city' else 6,10) <= P.fatigue:
                output("You were unable to keep up with training.",'red')
                P.takeAction()
                Train(abilities, master, False)
            elif lvl < 8:
                output(f"You successfully leveled up in {abilities}",'green')
                P.updateSkill(abilities,1)
                exitActionLoop(None,1)()
            elif rbtwn(1,4) == 1:
                output(f"You successfully leveled up in {abilities}",'green')
                P.updateSkill(abilities,1)
                exitActionLoop(None,1)()
            else:
                critthink = P.activateSkill('Critical Thinking')
                if rbtwn(1,16) <= critthink:
                    P.useSkill('Critical Thinking', 2, 7)
                    output(f"You successfully leveled up in {abilities}",'green')
                    P.updateSkill(abilities,1)
                    exitActionLoop(None,1)()
                else:
                    output("You were unsuccessful.",'red')
                    P.takeAction()
                    Train(abilities, master, False)

# Consequences
def C_road(action=1):
    P = lclPlayer()
    coord, depth = P.currenttile.findNearest(cities)
    r = np.random.rand()
    if r <= (depth/6):
        encounter('Highway Robber',[3,30],['Physical','Trooper'],{'coins':[0,3]}, consume='road', action_amt=action)
    elif P.currenttile.trader_rounds == 0:
        r = rbtwn(1,6)
        if r == 1:
            # Trader appears with 1/6 chance
            P.currenttile.trader_appears()
        exitActionLoop('road')()
    
def C_plains():
    P = lclPlayer()
    survival = P.activateSkill("Survival")
    r = rbtwn(0,8)
    if r <= survival:
        P.useSkill('Survival')
        output('Successfully avoided traps','green')
        exitActionLoop()()
    else:
        output('Trap hits you, +1 fatigue', 'red')
        paralyzed = P.takeDamage(0,1)
        if not paralyzed: exitActionLoop()()
    
    
def C_cave(tier=1, skip=False):
    P = lclPlayer()
    def enemy_approach(_=None):
        stealth = P.activateSkill('Stealth')
        r = rbtwn(0,4*tier)
        if r <= stealth:
            P.useSkill('Stealth',1,2*tier-1)
            output('Successfully avoided monster','green')
            exitActionLoop(f'tiered {tier}')()
        else:
            rewards = {1: {'hide':1, 'raw meat':1}, 2: {'hide':3, 'raw meat':1}, 3: {'hide':5, 'raw meat':2}}
            encounter('Monster',[[None,3,15,35][tier], [None,20,40,60][tier]], ['Physical','Wizard','Elemental','Trooper'],rewards[tier], consume=f'tiered {tier}')
    if not skip:
        fainted = False
        survival = P.activateSkill('Survival')
        r = rbtwn(0,2**tier)
        if r <= survival:
            P.useSkill('Survival',1,2*tier-1)
            output('Successfully evaded the sharp rocks','green')
        else:
            hp_dmg = 1 + (2*(tier-1))
            output(f'Fell on the harsh rocks: -{hp_dmg} HP','red')
            fainted = P.takeDamage(hp_dmg, 0)
        if not fainted: enemy_approach()
    else:
        return enemy_approach
    
def C_outpost(skip=False):
    def enemy_approach(_=None):
        P = lclPlayer()
        stealth = P.activateSkill('Stealth')
        r = rbtwn(0,12)
        if r <= stealth:
            P.useSkill('Stealth')
            output('Successfully avoided bandit','green')
            exitActionLoop()()
        else:
            encounter('Bandit',[15,40],['Physical','Elemental'],{'coins':[2,5]})
    if not skip:
        r = rbtwn(1,3)
        if r==1:
            enemy_approach()
        else:
            exitActionLoop()()
    else:
        return enemy_approach
    
def C_mountain(tier=1):
    P = lclPlayer()
    survival = P.activateSkill('Survival')
    r = rbtwn(0,2**tier)
    fainted_or_paralyzed = False
    if r <= survival:
        P.useSkill('Survival',1,2*tier-1)
        output('Successfully traversed harsh environment','green')
    else:
        output(f'Buffeted by harsh environment: -{tier} HP, -{tier} fatigue','red')
        fainted_or_paralyzed = P.takeDamage(tier, tier)
    if (not fainted_or_paralyzed): exitActionLoop(f'tiered {tier}')()
    
def C_oldlibrary(skip=False):
    def hermit_approach(_=None):
        P = lclPlayer()
        persuasion = P.activateSkill('Persuasion')
        r = rbtwn(1,12)
        if r <= persuasion:
            P.useSkill('Persuasion')
            output('Successfully persuaded the hermit to teach you!','green')
            Train(['Cunning','Critical Thinking'],'hermit',False)
        else:
            output('Unsuccessful in persuasion','red')
            encounter('Hermit',[55,75],['Wizard','Elemental'],{'coins':[5, 9]},{'Hit Points':12})
    if not skip:
        r = rbtwn(1,4)
        if r == 1:
            hermit_approach()
        else:
            exitActionLoop()()
    else:
        return hermit_approach
    
def C_ruins(skip=False, ret_func=False):
    P = lclPlayer()
    def spring_trap(_=None):
        if P.paused:
            return
        fainted = False
        survival = P.activateSkill('Survival')
        r = rbtwn(1,12)
        if r <= survival:
            P.useSkill('Survival')
            output('Successfully avoided traps','green')
        else:
            output('Trap hits you take 2 HP damage','red')
            fainted = P.takeDamage(2, 0)
        if ret_func=='trap':
            if not fainted: exitActionLoop()()
    def get_cloth(_=None):
        city = P.currenttile.neighbortiles.intersection({'fodker','zinzibar','enfeir'}).pop()
        cloth = f'old {city} cloth'
        if ret_func == 'cloth':
            getItem(cloth)()
        else:
            return cloth
    def approach(_=None):
        if P.paused:
            return
        persuasion = P.activateSkill('Persuasion')
        r = rbtwn(1,12)
        if r <= persuasion:
            P.useSkill('Persuasion')
            output("Successfully persuaded ancient wizard to teach you",'green')
            Train(['Hit Points','Heating'],'ancient wizard',False)
        else:
            output("Unsuccessful in persuasion",'red')
            # The ruins must be a neighbor of one of the three cities, the reward depends on that.
            encounter('Ancient Wizard',[60,85],['Wizard'],{get_cloth():2},{'Cunning':12})
    if not skip:
        spring_trap()
        approach()
    elif ret_func == 'trap':
        return spring_trap
    elif ret_func == 'cloth':
        return get_cloth
    else:
        return approach
    
def C_battlezone(skip=False):
    P = lclPlayer()
    def ninja_encounter(_=None):
        r = rbtwn(1,4)
        if r == 1:
            encounter('Ninja',[65,90],['Physical'],{'coins':[7,11]},{'Stealth':[6,8]})
        elif r == 2:
            encounter('Ninja',[45,70],['Physical'],{'coins':[5,9]},{'Stealth':[5,6]})
        elif r == 3:
            encounter('Ninja',[35,60],['Physical'],{'coins':[8,15]},{'Stealth':5},2)
        else:
            encounter('Ninja',[25,50],['Physical'],{'coins':[6,17]},{'Stealth':3},3)
    if not skip:
        stealth = P.activateSkill('Stealth')
        r = rbtwn(1,16)
        if r <= stealth:
            P.useSkill('Stealth',2,7)
            output("Avoided ninjas.",'green')
            exitActionLoop()()
        else:
            ninja_encounter()
    else:
        return ninja_encounter

def C_wilderness():
    P = lclPlayer()
    survival = P.activateSkill("Survival")
    r = rbtwn(1,14)
    if r <= survival:
        P.useSkill("Survival",2,7)
        output("Survived difficult terrain",'green')
        exitActionLoop()()
    else:
        output("Took 3 fatigue and 2 HP damage from terrain",'red')
        fainted_or_paralyzed = P.takeDamage(2, 3)
        if not fainted_or_paralyzed:
            stealth = P.activateSkill("Stealth")
            r = rbtwn(1,14)
            if r <= stealth:
                P.useSkill("Stealth",2,7)
                output("Avoided dangerous wilderness monster",'green')
                exitActionLoop()()
            else:
                encounter('Wild Vine Monster',[95,120],['Elemental'],{'bark':4})

consequences = {'road':C_road, 'plains':C_plains, 'cave':C_cave, 'outpost':C_outpost, 'mountain':C_mountain, 'oldlibrary':C_oldlibrary, 'ruins':C_ruins, 'battle1':C_battlezone, 'battle2':C_battlezone, 'wilderness':C_wilderness}

# Actions
KnowledgeBooks = {'critical thinking book':1,'bartering book':2,'persuasion book':2,'crafting book':2,
                  'heating book':3,'smithing book':2,'stealth book':1,'survival book':3,'gathering book':2,'excavating book':2}
def getBook(rarity=True):
    def getbook(_=None):
        L = list(KnowledgeBooks)
        if rarity:
            cumsum = [0]
            for book in L:
                cumsum.append(KnowledgeBooks[book]+cumsum[-1])
            r, i = rbtwn(1,cumsum[-1]), 1
            while (r > cumsum[i]):
                i += 1
            return L[i-1]
        else:
            return np.random.choice(L)
    return getbook
def book2skill(book):
    wl = book.split(' ')
    for i in range(len(wl)):
        wl[i] = wl[i][0].upper() + wl[i][1:]
    return ' '.join(wl[:-1])

def Gather(item, discrete):
    P = lclPlayer()
    def gather(_=None):
        if P.paused:
            return
        gathering = P.activateSkill('Gathering')
        gathered = 1
        while (gathered<6) and (rbtwn(1,12)<=gathering):
            if gathered == 1:
                P.useSkill('Gathering')
            gathered += 1
        output(f"Gathered {gathered} {item}",'green')
        P.addItem(item, gathered)
        exitActionLoop(None, 1)()
    return gather

def Excavate(result, max_result):
    P = lclPlayer()
    def excavate(_=None):
        if P.paused:
            return
        excavating = P.activateSkill('Excavating')
        rs = rbtwn(1,max_result,excavating+1)
        actions = {}
        for r in rs:
            for lbl, val in result.items():
                if lbl in actions:
                    continue
                if isbetween(val[0], val[1], r):
                    actions[lbl] = val[2] # Get the action button ready for this lbl
        if len(actions) == 0:
            output("Failed to find anything",'red')
            exitActionLoop(None,1,True)()
        elif len(actions) == 1:
            for item, func in actions.items():
                output(f"Excavation found {item}")
                func()
        else:
            output("Excavated the following (choose 1):")
            actionGrid(actions, False)
    return excavate

def persuade_trainer(abilities, master, fail_func, threshold=12):
    P = lclPlayer()
    def persuade_teacher(_=None):
        if P.paused:
            return
        persuasion = P.activateSkill('Persuasion')
        if rbtwn(1,threshold) <= persuasion:
            P.useSkill('Persuasion')
            output(f'Persuaded {master} to teach you.','green')
            Train(abilities, master, False)
        else:
            output(f'Failed to persuade {master} to teach you.','red')
            fail_func()
    return persuade_teacher

def A_road():
    P = lclPlayer()
    if P.currenttile.trader_rounds > 0:
        actions = {'Trader': await_trading(P.currenttile.trader_wares, True, False)}
        actionGrid(actions, True)
    else:
        # Otherwise clear the action grid except rest stuff
        actionGrid({}, True)

def A_plains():
    exc = {'Hunstman':[1,1,persuade_trainer(['Agility','Gathering'],'huntsman',exitActionLoop())],
           'Wild Herd':[2,6,Gather('raw meat',0)]}
    actions = {'Excavate':Excavate(exc,9)}
    actionGrid(actions, True)
    
def A_pond():
    exc = {'Go Fishing':[1,5,Gather('raw fish',0)],
           'Clay':[6,8,getItem('clay')],
           'Giant Serpent':[9,9,encounter('Giant Serpent',[20,45],['Elemental'],{'scales':[1,3]})]}
    actions = {'Excavate':Excavate(exc,12)}
    actionGrid(actions, True)
    
def A_cave(tier=1):
    def move_down(_):
        P = lclPlayer()
        if P.paused:
            return
        C_cave(tier+1, False)
    def move_up(_):
        P = lclPlayer()
        if P.paused:
            return
        C_cave(tier-1, False)
    exc = {1:{'Lead':[1,4,getItem('lead')],'Tin':[5,8,getItem('tin')],'Monster':[9,15,C_cave(1,True)]},
           2:{'Tantalum':[1,3,getItem('tantalum')],'Aluminum':[4,6,getItem('aluminum')],'Monster':[7,14,C_cave(2,True)]},
           3:{'Tungsten':[1,2,getItem('tungsten')],'Titanium':[3,4,getItem('titanium')],'Monster':[5,13,C_cave(3,True)]}}
    action_tiers = {1:{'Excavate':Excavate(exc[1],20),'Descend':move_down},
                    2:{'Excavate':Excavate(exc[2],20),'Descend':move_down,'Ascend':move_up},
                    3:{'Excavate':Excavate(exc[3],20),'Ascend':move_up}}
    actionGrid(action_tiers[tier], True)
            
def A_outpost():
    exc = {'String':[1,3,getItem('string')],
           'Beads':[4,6,getItem('beads')],
           'Sand':[7,8,getItem('sand')],
           'Bandit':[9,10,C_outpost(True)]}
    actions = {'Excavate':Excavate(exc,12)}
    actionGrid(actions, True)
    
def A_mountain(tier=1):
    def move_down(_):
        P = lclPlayer()
        if P.paused:
            return
        C_mountain(tier-1)
    def move_up(_):
        P = lclPlayer()
        if P.paused:
            return
        C_mountain(tier+1)
    exc = {1:{'Copper':[1,5,getItem('copper')],'Iron':[6,10,getItem('iron')],'Monk':[11,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           2:{'Kevlium':[1,4,getItem('kevlium')],'Nickel':[5,8,getItem('nickel')],'Monk':[9,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           3:{'Diamond':[1,3,getItem('tungsten')],'Chromium':[4,6,getItem('titanium')],'Monk':[7,8,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]}}
    action_tiers = {1:{'Excavate':Excavate(exc[1],20),'Ascend':move_up},
                    2:{'Excavate':Excavate(exc[2],20),'Descend':move_down,'Ascend':move_up},
                    3:{'Excavate':Excavate(exc[3],20),'Descend':move_down}}
    actionGrid(action_tiers[tier], True)
    
def A_oldlibrary():
    exc = {'Book':[1,8,getBook()],'Hermit':[9,14,C_oldlibrary(True)]}
    actions = {'Excavate':Excavate(exc,20)}
    actionGrid(actions, True)
    
def A_ruins():
    def read_attempt(_):
        P = lclPlayer()
        book = getBook(True)()
        skill = book2skill(book)
        if P.skills[skill] < 8:
            critthink = P.activateSkill('Critical Thinking')
            r = rbtwn(1,12)
            if r <= critthink:
                output(f"Understood teachings from old {book}.",'green')
                P.updateSkill(skill)
            else:
                # Get 3 xp from reading
                output(f"Understood some from old {book}.")
                P.addXP(skill, 3)
        else:
            output(f"Found old {book} but can't learn anything new.",'yellow')
        exitActionLoop()()
    exc = {'Trap':[1,5,C_ruins(True,'trap')],
           'Cloth':[6,7,C_ruins(True,'cloth')],
           'Wizard':[8,10,C_ruins(True,'ancient wizard')],
           'Old Book':[11,14,read_attempt]}
    actions = {'Excavate':Excavate(exc,20)}
    actionGrid(actions, True)
    
def A_battlezone():
    def rare_find(_):
        r = rbtwn(1,4)
        if r==1:
            getItem('shinopsis')()
        elif r==2:
            getItem('ebony')()
        elif r==3:
            getItem('astatine')()
        else:
            getItem('promethium')()
    exc = {'Tantalum':[1,1,getItem('tantalum')],
           'Aluminum':[2,2,getItem('aluminum')],
           'Kevlium':[3,3,getItem('kevlium')],
           'Nickel':[4,4,getItem('nickel')],
           'Tungsten':[5,5,getItem('tungsten')],
           'Titanium':[6,6,getItem('titanium')],
           'Diamond':[7,7,getItem('diamond')],
           'Chromium':[8,8,getItem('chromium')],
           'Rare':[9,9,rare_find],
           'Ninja':[10,16,C_battlezone(True)],
           'Nothing':[17,20,exitActionLoop()]}
    actions = {'Excavate':Excavate(exc,20)}
    actionGrid(actions, True)
    
def A_wilderness():
    def rare_find(_):
        r = rbtwn(1,6)
        if r==1:
            getItem('shinopsis', empty_tile=True)()
        elif r==2:
            getItem('ebony', empty_tile=True)()
        elif r==3:
            getItem('astatine', empty_tile=True)()
        elif r==4:
            getItem('promethium', empty_tile=True)()
        else:
            getItem('gem', empty_tile=True)()
    def tree_gather(_):
        P = lclPlayer()
        gathering = P.activateSkill("Gathering")
        if gathering > 0:
            P.useSkill("Gathering", 2, 7)
            item = 'fruit' if rbtwn(1,2)==1 else 'bark'
            output(f"You gather {gathering} {item}")
            getItem(item, gathering, empty_tile=True)
        else:
            exitActionLoop(empty_tile=True)()
    exc = {'Rare Find':[1,3,rare_find],'Grand Tree':[4,10,tree_gather],'Nothing':[11,20,exitActionLoop(empty_tile=True)]}
    actions = {'Excavate':Excavate(exc,20)}
    actionGrid(actions, True)

avail_actions = {'road':A_road,'plains':A_plains,'pond':A_pond,'cave':A_cave,'outpost':A_outpost,'mountain':A_mountain,'oldlibrary':A_oldlibrary,'ruins':A_ruins,'battle1':A_battlezone,'battle2':A_battlezone}#,'wilderness','village1','village2','village3','village4','village5',
#                 'anafola','benfriege','demetry','enfeir','fodker','glaser','kubani','pafiz','scetcher','starfex','tamarania','tamariza','tutalu','zinzibar'}


capital_info = {'anafola':{'home':40,'home_cap':5,'capacity':4,'market':20,'return':5,'market_cap':2,'invest':8},
                'benfriege':{'home':8,'home_cap':2,'capacity':4,'market':3,'return':1,'market_cap':1,'invest':3},
                'demetry':{'home':49,'home_cap':6,'capacity':4,'market':24,'return':6,'market_cap':2,'invest':9},
                'enfeir':{'home':20,'home_cap':4,'capacity':4,'market':9,'return':2,'market_cap':1,'invest':3},
                'fodker':{'home':24,'home_cap':4,'capacity':4,'market':12,'return':3,'market_cap':1,'invest':None}, # Fodker has no villages to invest in
                'glaser':{'home':5,'home_cap':2,'capacity':4,'market':3,'return':1,'market_cap':1,'invest':3},
                'kubani':{'home':37,'home_cap':5,'capacity':4,'market':19,'return':4,'market_cap':2,'invest':7},
                'pafiz':{'home':27,'home_cap':4,'capacity':4,'market':13,'return':3,'market_cap':1,'invest':5},
                'scetcher':{'home':42,'home_cap':5,'capacity':4,'market':20,'return':5,'market_cap':2,'invest':8},
                'starfex':{'home':28,'home_cap':4,'capacity':4,'market':14,'return':3,'market_cap':1,'invest':5},
                'tamarania':{'home':43,'home_cap':5,'capacity':4,'market':21,'return':5,'market_cap':2,'invest':8},
                'tamariza':{'home':42,'home_cap':5,'capacity':4,'market':21,'return':5,'market_cap':2,'invest':8},
                'tutalu':{'home':23,'home_cap':4,'capacity':4,'market':10,'return':3,'market_cap':1,'invest':4},
                'zinzibar':{'home':8,'home_cap':2,'capacity':4,'market':2,'return':1,'market_cap':1,'invest':3}}

class Player(Image):
    def __init__(self, board, username, birthcity, **kwargs):
        super().__init__(**kwargs)
        self.source = f'images\\characters\\{birthcity}.png'
        self.parentBoard = board
        self.username = username
        self.birthcity = birthcity
        self.imgSize = pilImage.open(self.source).size
        
        # Player Victory Points
        self.Combat = 3
        self.Capital = 0
        self.Reputation = 0
        self.Knowledge = 0
        
        # Constraints
        self.paused = False
        self.max_road_moves = 2
        self.max_actions = 2
        self.max_avail_transactions = 3
        self.max_capacity = 3
        self.max_fatigue = 10
        self.combatstyle = cities[self.birthcity]['Combat Style']
        
        # Current Stats
        self.road_moves = 2
        self.actions = 2
        self.item_count = 0
        self.items = {}
        self.activated_bartering = False
        self.avail_transactions = 3
        self.fatigue = 0
        
        self.coins = cities[self.birthcity]['Coins']
        self.paralyzed_rounds = 0
        self.tiered = False
        
        # Player Track
        #Combat
        self.attributes = {'Agility':0,'Cunning':1,'Technique':2,'Hit Points':3,'Attack':4,'Stability':5,'Def-Physical':6,'Def-Wizard':7, 'Def-Elemental':8, 'Def-Trooper':9}
        self.atrorder = ['Agility','Cunning','Technique','Hit Points','Attack','Stability','Def-Physical','Def-Wizard','Def-Elemental','Def-Trooper']
        self.combat = np.array([0, 0, 0, 2, 1, 0, 0, 0, 0, 0])
        self.boosts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for atr, val in cities[self.birthcity]['Combat Boosts']:
            self.boosts[self.attributes[atr]] += val
        self.current = self.combat + self.boosts
        #Knowledge
        self.skills = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
        self.xps = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
        for skl, val in cities[self.birthcity]['Knowledges']:
            self.updateSkill(skl, val)
        #Capital
        self.cityorder = sorted(cities)
        self.homes = {city: False for city in cities}
        self.markets = {city: False for city in cities}
        self.bank = {city: 0 for city in cities}
        self.villages = {city: {} for city in cities}
        self.awaiting = {city: '' for city in cities}
        # Some helpers of Capital
        self.already_asset_bartered = False
        self.training_allowed = {city: False for city in cities}
        self.training_allowed[self.birthcity] = True
        self.market_allowed = {city: False for city in cities}
        self.market_allowed[self.birthcity] = True
        # Reputation
        self.reputation = np.empty((8, 5),dtype=object)
        self.entry_allowed = {city: False for city in cities}
        self.entry_allowed[self.birthcity] = True
        # Position Player
        self.currentcoord = positions[birthcity][0]
        self.currenttile = self.parentBoard.gridtiles[self.currentcoord]
        self.moveto(self.currentcoord, False, True)
        self.size_hint = (self.imgSize[0]/xsize, self.imgSize[1]/ysize)
    def updateView(self):
        self.size_hint = (self.imgSize[0]*(self.parentBoard.zoom+1)/xsize, self.imgSize[1]*(self.parentBoard.zoom+1)/ysize)
        self.pos_hint = {'center_x':self.currenttile.centx, 'center_y':self.currenttile.centy}
    def get_recover(self):
        if (self.currenttile.tile in self.homes) and self.homes[self.currenttile.tile]:
            return 4
        elif self.currenttile.tile == self.birthcity:
            return 2
        return 1
    def go2consequence(self):
        if self.currenttile.tile in consequences:
            consequences[self.currenttile.tile]()
        else:
            exitActionLoop(None,1)()
    def go2action(self, param=None):
        if (self.currenttile.tile in avail_actions) and (not self.currenttile.is_empty):
            if param is None:
                avail_actions[self.currenttile.tile]()
            else:
                avail_actions[self.currenttile.tile](param)
        else:
            actionGrid({}, True)
    def moveto(self, coord, trigger_consequence=True, skip_check=False):
        nxt = self.parentBoard.gridtiles[coord]
        if nxt.tile in cities:
            if not self.entry_allowed[nxt.tile]:
                output("The city will not let you enter! (Gain some reputation!)",'red')
                return
        if (not skip_check) and (self.currenttile.tile != 'road') and (self.currenttile.tile not in cities) and (nxt.tile=='road'):
            output("Move there on same action (+2 Fatigue)?")
            actionGrid({'Yes':(lambda _:self.moveto(coord,trigger_consequence,3)),'No':(lambda _:self.moveto(coord,trigger_consequence,2))}, False)
        else:
            if skip_check != True:
                socket_client.send('[MOVE]',coord)
            self.currentcoord = coord
            self.currenttile = self.parentBoard.gridtiles[coord]
            game_app.game_page.recover_button.text = f'Rest ({self.get_recover()})'
            self.parentBoard.updateView()
            self.pos_hint = {'center_x':self.currenttile.centx, 'center_y':self.currenttile.centy}
            # Take damage and action afterwards
            if skip_check == 3:
                self.takeDamage(0,2)
                self.update_mainStatPage()
            elif skip_check == 2:
                output(f"[ACTION {self.max_actions-self.actions+1}] Moving onto road consumes action.")
                self.actions -= 1
                self.takeDamage(0,1)
                self.update_mainStatPage()
            if trigger_consequence: self.go2consequence()
    def add_PlayerTrack(self):
        self.PlayerTrack = PlayerTrack(self)
        screen = Screen(name='Player Track')
        screen.add_widget(self.PlayerTrack)
        self.parentBoard.game_page.main_screen.add_widget(screen)
    def get_mainStatUpdate(self):
        return [f'{self.fatigue}/{self.max_fatigue}',
                f'{self.current[self.attributes["Hit Points"]]}/{self.combat[self.attributes["Hit Points"]]+self.boosts[self.attributes["Hit Points"]]}',
                f'{self.coins}',
                f'{self.item_count}/{self.max_capacity}']
    def add_mainStatPage(self):
        self.mtable = Table(['Fatigue','HP','Coins','Items'],[self.get_mainStatUpdate()],False,(1,1,1,0.5),text_color=(0,0,0,0.9),
                            pos_hint={'x':0,'y':game_app.game_page.stat_ypos}, size_hint_y=game_app.game_page.stat_ysize)
        game_app.game_page.right_line.add_widget(self.mtable)
    def update_mainStatPage(self):
        updated_data = self.get_mainStatUpdate()
        for i in range(len(updated_data)):
            self.mtable.cells[self.mtable.header[i]][0].text = updated_data[i]
        self.PlayerTrack.updateAll()
    def recover(self, _):
        if not self.paused:
            rest_rate = self.get_recover()
            self.fatigue = max([0, self.fatigue-rest_rate])
            hp_idx = self.attributes['Hit Points']
            self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+rest_rate])
            output(f'[ACTION {self.max_actions-self.actions+1}] You rested {rest_rate} fatigue/HP')
            self.takeAction(0, False)
    def eat(self, food=None, ftg=0, hp=0):
        def eat_food(_):
            # Prevent eating when game is paused just in case they are in a fight
            if not self.paused:
                restoring_list = []
                if ftg>0: restoring_list.append(f'fatigue by {ftg}')
                if hp>0: restoring_list.append(f'HP by {hp}')
                output(f'Ate {food} restoring {", ".join(restoring_list)}')
                self.rmvItem(food)
                self.fatigue = max([0, self.fatigue - ftg])
                hp_idx = self.attributes['Hit Points']
                self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+hp])
                self.update_mainStatPage()
                # Then restore back to original action list
                self.go2action()
        if food is None:
            def choose_food(_):
                # Prevent eating when game is paused just in case they are in a fight
                if not self.paused:
                    food_func = {'raw meat':self.eat('raw meat',1,0),'cooked meat':self.eat('cooked meat',2,0),'well cooked meat':self.eat('well cooked meat',3,0),
                                 'raw fish':self.eat('raw fish',0,1),'cooked fish':self.eat('cooked fish',0,2),'well cooked fish':self.eat('well cooked fish',0,3),
                                 'fruit':self.eat('fruit',1,1)}
                    try_to_eat = {}
                    for food in food_func:
                        if food in self.items:
                            try_to_eat[food] = food_func[food]
                    if len(try_to_eat)==0:
                        output('You have no food to eat!','yellow')
                    elif len(try_to_eat)==1:
                        for func in try_to_eat.values():
                            func(None)
                    else:
                        actionGrid(try_to_eat, False)
            return choose_food
        else:
            return eat_food
    def pause(self):
        self.paused = True
        if self == self.parentBoard.localPlayer:
            output("Round End")
        if len(self.parentBoard.Players) != len(game_app.launch_page.usernames):
            # If not all players have been created then do not end the round yet
            return
        for P in self.parentBoard.Players.values():
            if not P.paused:
                # If someone has not ended the round yet, then wait.
                return
        # Otherwise start the round
        self.startRound()
    def startRound(self):
        for P in self.parentBoard.Players.values():
            P.paused = False
        self.parentBoard.localPlayer.actions = self.max_actions
        self.parentBoard.sendFaceMessage('Start Round!')
        for T in self.parentBoard.gridtiles.values():
            T.update_tile_properties()
        check_paralysis()
    def takeDamage(self, hp_amt, ftg_amt, add=True):
        hp_idx = self.attributes['Hit Points']
        self.current[hp_idx] = max([0, self.current[hp_idx]-hp_amt])
        fainted = False
        if self.current[hp_idx] == 0:
            if self.Combat > 2:
                CL = list(self.attributes)
                while True:
                    Ratr = np.random.choice(CL)
                    if ((Ratr=='Attack') or (Ratr=='Hit Points')) and (self.combat[self.attributes[Ratr]]>1):
                        break
                    elif self.combat[self.attributes[Ratr]] > 0:
                        break
                self.updateAttribute(Ratr, -1)
                self.parentBoard.sendFaceMessage('You Fainted!','red')
                output(f"You lose {Ratr}, are paralyzed, and rushed to nearest city.",'red')
            # Just in case they die in a tiered environment, make sure they are not stuck.
            self.tiered = False
            # Move to nearest city
            coord, distance = self.currenttile.findNearest(cities)
            self.moveto(coord)
            # Artificially paralyze the player by setting fatigue to 11
            self.fatigue = 11
            fainted = True
        if not fainted:
            self.fatigue = self.fatigue + ftg_amt if add else ftg_amt
        paralyzed = True if self.fatigue>10 else False
        check_paralysis()
        return (fainted or paralyzed)
    def takeAction(self, fatigue=1, verbose=True):
        if verbose: output(f"[ACTION {self.max_actions-self.actions+1}] You took {fatigue} fatigue")
        self.actions -= 1
        if self.activated_bartering:
            # Player performed an action after bartering - make sure you give them an extra fatigue if they bartered and didn't do 3 transactions
            fatigue += 1
            self.activated_bartering = False
        if fatigue > 0: self.takeDamage(0,fatigue)
        self.road_moves = self.max_road_moves # If action is taken, then the road moves should be refreshed.
        self.already_asset_bartered = False # If an action is taken, you can try to barter for the asset again.
        self.update_mainStatPage()
        if self.actions <= 0:
            self.pause()
            socket_client.send("[ROUND]",'end')
    def updateSkill(self, skill, val=1):
        self.skills[skill] += val
        self.Knowledge += val
    def addXP(self, skill, xp, msg=''):
        output(f"{msg}Gained {xp}xp for {skill}.",'green')
        self.xps[skill] += xp
        while (self.xps[skill] >= (3 + self.skills[skill])):
            self.xps[skill] -= (3 + self.skills[skill])
            self.updateSkill(skill)
            output(f"Leveled up {skill} to {self.skills[skill]}!",'green')
    def activateSkill(self, skill, xp=1, max_lvl_xp=2):
        lvl = self.skills[skill]
        if rbtwn(1,10) <= self.fatigue:
            output(f"Fatigue impacted your {skill} skill.",'yellow')
            actv_lvl = np.max([0,lvl-self.fatigue])
        else:
            actv_lvl = lvl
            if lvl <= max_lvl_xp: self.addXP(skill, xp, 'Activation: ')
        return actv_lvl
    def useSkill(self, skill, xp=1, max_lvl_xp=5):
        if self.skills[skill] <= max_lvl_xp:
            self.addXP(skill, xp, 'Successful: ')
    def updateAttribute(self, attribute, val=1):
        self.combat[self.attributes[attribute]] += val
        self.current[self.attributes[attribute]] += val
        self.Combat += val
        self.update_mainStatPage()
    def levelup(self, ability, val=1):
        if ability in self.skills:
            self.updateSkill(ability, val)
        else:
            self.updateAttribute(ability, val)
    def get_level(self, ability):
        if ability in self.skills:
            return self.skills[ability]
        return self.combat[self.attributes[ability]]
    def purchase(self, asset, city, barter=None, _=None):
        if self.paused:
            return
        cost = capital_info[city][asset]
        def make_purchase(asset, city, cost, _=None):
            if self.paused:
                return
            if self.coins < cost:
                output('Insufficient funds!','yellow')
                exitActionLoop(amt=0)()
                return
            if asset == 'home':
                if self.homes[city]:
                    output('You already own a home here!','yellow')
                    exitActionLoop(amt=0)()
                    return
                self.homes[city] = True
                self.max_capacity += capital_info[city]['capacity']
                self.training_allowed[city] = True
                self.market_allowed[city] = True
                self.Capital += capital_info[city]['home_cap']
            else:
                if self.markets[city]:
                    output('You already own a market here!','yellow')
                    exitActionLoop(amt=0)()
                    return
                self.markets[city] = True
                self.market_allowed[city] = True
                self.Capital += capital_info[city]['market_cap']
            exitActionLoop()()
        if barter is None:
            output(f"This will cost {cost}. Attempt to Barter (+1 Fatigue)?", 'blue')
            actions = {'Yes':partial(self.purchase, asset, city, True), 'No':partial(self.purchase, asset, city, False)}
            actionGrid(actions, False)
        elif barter and (not self.already_asset_bartered):
            if self.coins < cost:
                output("You can't effectively barter when your funds are low!", 'yellow')
                exitActionLoop(amt=0)
                return
            self.already_asset_bartered = True
            bartering = self.activateSkill("Bartering")
            # Bartering home is more effective than bartering for market
            r = min([self.cost-1, rbtwn(0, bartering if asset == 'home' else bartering/2)]) # Has to at least cost a coin!
            output(f"Your bartering can save you {r} coins. Complete purchase?", 'blue')
            actions = {'Yes':partial(make_purchase, asset, city, cost-r), 'No':exitActionLoop(amt=0)}
            actionGrid(actions, False)
            self.fatigue += 1 # Take a fatigue for bartering
        elif barter and self.already_asset_bartered:
            output("You can't barter again this action!", 'yellow')
            return
        else:
            make_purchase(asset, city, cost)
    def rmvItem(self, item):
        if item not in self.items:
            output(f"Item {item} does not exist in inventory.",'yellow')
        else:
            self.items.pop(item)
    def addItem(self, item, amt):
        if self.item_count >= self.max_capacity:
            output("Cannot add anymore items!",'yellow')
            self.update_mainStatPage()
            return
        elif (self.item_count + amt) > self.max_capacity:
            output("Could not add all the items!",'yellow')
            amt = self.max_capacity - self.item_count
        self.item_count += amt
        if item in self.items:
            self.items[item] += amt
        else:
            self.items[item] = amt
        if self.items[item]==0:
            self.rmvItem(item)
        self.update_mainStatPage()

class HoverButton(Button, HoverBehavior):
    def __init__(self, display, message, **kwargs):
        super().__init__(**kwargs)
        self.display=display
        self.message=message
    def on_enter(self, *args):
        self.display.text = self.message

class Table(GridLayout):
    def __init__(self, header, data, wrap_text=True, bkg_color=None, header_color=None, text_color=None, super_header=None, header_height_hint=None, header_as_buttons=False, color_odd_rows=False,**kwargs):
        super().__init__(**kwargs)
        self.cols = len(header)
        self.wrap_text = wrap_text
        self.header = header
        self.color_odd_rows = (0.3, 1, 1, 0.5) if color_odd_rows and (type(color_odd_rows) is bool) else color_odd_rows
        if bkg_color is not None:
            with self.canvas.before:
                Color(bkg_color[0],bkg_color[1],bkg_color[2],bkg_color[3],mode='rgba')
                self.bkg = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self.update_bkgSize,size=self.update_bkgSize)
        if super_header is not None:
            super_kwargs = {}
            if text_color is not None: super_kwargs['color'] = text_color
            if header_height_hint is not None: 
                super_kwargs['height'] = Window.size[1]*header_height_hint
                super_kwargs['size_hint_y'] = None
            self.super_header = Label(text=super_header, markup=True, valign='bottom', halign='left', bold=True, **super_kwargs)
            self.add_widget(self.super_header)
            for i in range(len(header)-1):
                # Take up some random space to make sure the super header is on left and header starts as normal
                space_kwargs = {} if header_height_hint is None else {'height': Window.size[1]*header_height_hint, 'size_hint_y':None}
                self.add_widget(Widget(**space_kwargs))
        self.cells = {}
        for h in header:
            self.cells[h] = []
            clr = ['[b]','[/b]'] if header_color is None else [f'[color={get_hexcolor(header_color)}][b]','[/b][/color]']
            hkwargs = {} if text_color is None else {'color':text_color}
            if header_as_buttons:
                L = Button(text=clr[0]+h+clr[1],markup=True,background_color=(1,1,1,0),underline=True,**hkwargs)
            else:
                L = Label(text=clr[0]+h+clr[1],markup=True,valign='bottom',halign='center',**hkwargs)
                #L.text_size = L.size
                if wrap_text: L.text_size = L.size
            self.add_widget(L)
        # In the case that data is one-dimensional, then make it a matrix of one row.
        self.input_text_color = text_color
        self.wrap_text = wrap_text
        self.update_data_cells(data)
    def clear_data_cells(self):
        for h in self.cells:
            for L in self.cells[h]:
                L.text=''
                self.remove_widget(L)
    def update_data_cells(self, data, clear=True):
        if clear: self.clear_data_cells()
        data = np.reshape(data,(1,-1)) if len(np.shape(data))==1 else data
        i = 0
        for row in data:
            j = 0
            for item in row:
                cell = self.header[j]
                j += 1
                if not clear:
                    if type(item) is dict:
                        #print(cell, i, self.cells[cell][i].text, item['txt'])
                        self.cells[cell][i].text=item['text']
                    else:
                        self.cells[cell][i].text=str(item)
                    continue
                if type(item) is dict:
                    if (i % 2) and self.color_odd_rows:
                        old_bkg = item['background_color'] if 'background_color' in item else (1, 1, 1, 1)
                        item['background_color'] = [old_bkg[clri]*self.color_odd_rows[clri] for clri in range(len(old_bkg))]
                        if item['background_color'][-1] == 0: item['background_color'][-1] = self.color_odd_rows[-1]
                        item['background_color'] = tuple(item['background_color'])
                    func = None
                    if 'func' in item:
                        func = item['func']
                        item.pop('func')
                    if 'hover' in item:
                        display, message = item['hover']
                        item.pop('hover')
                        L = HoverButton(display, message, **item)
                    else:
                        L = Button(**item)
                    if func is not None:
                        L.bind(on_press=func)
                else:
                    L = Label(text=str(item)) if self.input_text_color is None else Label(text=str(item),color=self.input_text_color)
                    if self.wrap_text: L.text_size = L.size
                self.add_widget(L)
                self.cells[cell].append(L)
            i += 1
    def update_bkgSize(self, instance, value):
        self.bkg.size = self.size
        self.bkg.pos = self.pos
        if self.wrap_text:
            for L in self.children:
                L.text_size = L.size

class PlayerTrack(GridLayout):
    def __init__(self, player, **kwargs):
        super().__init__(**kwargs)
        self.player = player
        self.cols=1
        self.hclr = get_hexcolor((255, 85, 0))
        self.track_screen = ScreenManager()
        # Combat Screen
        self.Combat = Table(header=['Attribute','Base','Boost','Current'], data=self.get_Combat(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
        screen = Screen(name='Combat')
        screen.add_widget(self.Combat)
        self.track_screen.add_widget(screen)
        # Knowledge Screen
        self.Knowledge = Table(header=['Skill','Level', 'XP'], data=self.get_Knowledge(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
        screen = Screen(name='Knowledge')
        screen.add_widget(self.Knowledge)
        self.track_screen.add_widget(screen)
        # Capital Screen
        capGrid = GridLayout(cols=1)
        self.capitalDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0, 0, 0, 1),markup=True,background_color=(1,1,1,0))
        self.Capital = Table(header=['City','Home'], data=self.get_Capital(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50),header_as_buttons=True)#,'Market','Bank','Village1','Village2','Village3','Village4','Village5','Waiting']
        capGrid.add_widget(self.Capital)
        capGrid.add_widget(self.capitalDisplay)
        screen = Screen(name='Capital')
        screen.add_widget(capGrid)
        self.track_screen.add_widget(screen)
        # Reputation Screen
        # Item Screen
        self.Items = Table(header=['Item', 'Quantity', ''], data=self.get_Items(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50))
        screen = Screen(name='Items')
        screen.add_widget(self.Items)
        self.track_screen.add_widget(screen)
        # Add header tabs
        def changeTab(tabName,_):
            self.track_screen.current = tabName
        self.tabs = GridLayout(cols=5, height=0.05*Window.size[1], size_hint_y=None)
        self.tab_color = (0.1, 0.6, 0.5, 1)
        self.combatTab = Button(text=f"Combat: [color={self.hclr}]{player.Combat}[/color]",markup=True,background_color=self.tab_color)
        self.combatTab.bind(on_press=partial(changeTab, 'Combat'))
        self.knowledgeTab = Button(text=f'Knowledge: [color={self.hclr}]{player.Knowledge}[/color]',markup=True,background_color=self.tab_color)
        self.knowledgeTab.bind(on_press=partial(changeTab, 'Knowledge'))
        self.capitalTab = Button(text=f'Capital: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.capitalTab.bind(on_press=partial(changeTab, "Capital"))
        self.itemsTab = Button(text='Items: 0/3',markup=True,background_color=self.tab_color)
        self.itemsTab.bind(on_press=partial(changeTab, 'Items'))
        self.tabs.add_widget(self.combatTab)
        self.tabs.add_widget(self.knowledgeTab)
        self.tabs.add_widget(self.capitalTab)
        self.tabs.add_widget(self.itemsTab)
        # Add widgets in order
        self.add_widget(self.tabs)
        self.add_widget(self.track_screen)
    def get_Combat(self):
        data = []
        for i in range(len(self.player.atrorder)):
            data.append([{'text':str(self.player.atrorder[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.combat[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.boosts[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.current[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}])
        return data
    def get_Knowledge(self):
        data = []
        for skill in self.player.skills:
            data.append([{'text':skill,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.skills[skill]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.xps[skill]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}])
        return data
    def homeDisplay(self, city, _=None):
        data = {}
        if self.player.homes[city]:
            msg = f'Purchased! Capital: [color={self.hclr}]{capital_info[city]["home_cap"]}[/color], Capacity: {capital_info[city]["capacity"]}'
            data['disabled'] = True
            data['text'] = 'Owned'
            data['background_color'] = (1, 10, 1, 0.8)
        else:
            data['text'] = 'Buy'
            msg = f'Cost: [color=cfb53b]{capital_info[city]["home"]}[/color] Capital: [color={self.hclr}]{capital_info[city]["home_cap"]}[/color], Capacity: {capital_info[city]["capacity"]}'
            if self.player.currenttile.tile != city:
                data['disabled'] = True
            else:
                data['background_color'] = (1, 1, 0.2, 1)
                data['func'] = partial(self.player.purchase, 'home', city, None)
        data['hover'] = (self.capitalDisplay, msg)
        return data
    def get_Capital(self):
        data = []
        for city in self.player.cityorder:
            data.append([{'text':city[0].upper()+city[1:],'background_color':(1,1,1,0),'color':(0,0,0,1)}, self.homeDisplay(city)])
        return data
    def get_Items(self):
        data = []
        wares = self.player.currenttile.city_wares.union(self.player.currenttile.trader_wares)
        for item in self.player.items:
            sellbutton = {'text':'Sell', 'func':partial(sellItem, item, True if self.player.currenttile.trader_rounds>0 else False, None)} if item not in wares else {'text':'','background_color':(1,1,1,0),'disabled':False}
            data.append([{'text':item, 'background_color':(1,1,1,0), 'disabled':True,'color':(0,0,0,1)},
                         {'text':str(self.player.items[item]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         sellbutton])
            # [INCOMPLETE] Add an "invest" button option if player on a village
        return data
    def updateAll(self):
        self.Combat.update_data_cells(self.get_Combat(), False)
        self.combatTab.text=f"Combat: [color={self.hclr}]{self.player.Combat}[/color]"
        self.Knowledge.update_data_cells(self.get_Knowledge(), False)
        self.knowledgeTab.text=f'Knowledge: [color={self.hclr}]{self.player.Knowledge}[/color]'
        self.Capital.update_data_cells(self.get_Capital())
        self.capitalTab.text=f"Capital [color={self.hclr}]{self.player.Capital}[/color]"
        self.Items.update_data_cells(self.get_Items())
        self.itemsTab.text=f'Items: {self.player.item_count}/{self.player.max_capacity}'
        

cities = {'anafola':{'Combat Style':'Summoner','Coins':3,'Knowledges':[('Excavating',1),('Persuasion',1)],'Combat Boosts':[('Stability',2)]},
          'benfriege':{'Combat Style':'Elemental','Coins':2,'Knowledges':[('Crafting',2)],'Combat Boosts':[('Stability',1),('Cunning',1)]},
          'demetry':{'Combat Style':'Elemental','Coins':9,'Knowledges':[('Bartering',2)],'Combat Boosts':[]},
          'enfeir':{'Combat Style':'Physical','Coins':3,'Knowledges':[('Stealth',2)],'Combat Boosts':[('Cunning',2)]},
          'fodker':{'Combat Style':'Trooper','Coins':5,'Knowledges':[('Smithing',1)],'Combat Boosts':[('Stability',2),('Def-Physical',2)]},
          'glaser':{'Combat Style':'Elemental','Coins':2,'Knowledges':[('Survival',1)],'Combat Boosts':[('Def-Trooper',3),('Cunning',1)]},
          'kubani':{'Combat Style':'Physical','Coins':3,'Knowledges':[('Crafting',1),('Gathering',1)],'Combat Boosts':[('Def-Wizard',3),('Agility',1)]},
          'pafiz':{'Combat Style':'Wizard','Coins':4,'Knowledges':[('Persuasion',1)],'Combat Boosts':[('Def-Elemental',3)]},
          'scetcher':{'Combat Style':'Physical','Coins':4,'Knowledges':[],'Combat Boosts':[('Attack',1),('Hit Points',1),('Def-Physical',1),('Def-Wizard',1),('Def-Elemental',1),('Def-Trooper',1)]},
          'starfex':{'Combat Style':'Elemental','Coins':5,'Knowledges':[('Heating',1),('Gathering',1)],'Combat Boosts':[('Attack',2)]},
          'tamarania':{'Combat Style':'Physical','Coins':7,'Knowledges':[('Smithing',2)],'Combat Boosts':[('Attack',2)]},
          'tamariza':{'Combat Style':'Wizard','Coins':5,'Knowledges':[('Critical Thinking',1)],'Combat Boosts':[('Def-Physical',1),('Def-Elemental',2),('Hit Points',1)]},
          'tutalu':{'Combat Style':'Trooper','Coins':3,'Knowledges':[('Excavating',2)],'Combat Boosts':[('Attack',3)]},
          'zinzibar':{'Combat Style':'Physical','Coins':2,'Knowledges':[('Stealth',2)],'Combat Boosts':[('Agility',2),('Attack',1)]}}
sellPrice = {1:0, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:5, 9:6}
mrktPrice = {1:1, 2:3, 3:5, 4:6, 5:8, 6:9, 7:10, 8:12, 9:20}
gameItems = {'Food':{'raw meat':1,'cooked meat':2,'well cooked meat':3,'raw fish':1,'cooked fish':2,'well cooked fish':3,'fruit':2},
             'Crafting':{'string':1,'beads':1,'hide':1,'sand':1,'clay':2,'scales':2,'leather':2,'bark':2,'ceramic':3,'glass':5,'rubber':6,'gems':8},
             'Smithing':{'lead':1,'tin':1,'copper':1,'iron':1,'tantalum':3,'aluminum':3,'kevlium':3,'nickel':3,'tungsten':6,'titanium':6,'diamond':6,'chromium':6,'shinopsis':9,'ebony':9,'astatine':9,'promethium':9},
             'Knowledge Books':{'critical thinking book':8,'bartering book':5,'persuasion book':3,'heating book':4,'smithing book':4,'stealth book':9,'survival book':6,'gathering book':5,'excavating book':8},
             'Cloth':{'benfriege cloth':4,'enfeir cloth':2,'glaser cloth':5,'pafiz cloth':2,'scetcher cloth':2,'starfex cloth':2,'tutalu cloth':2,'zinzibar cloth':2,'old fodker cloth':6,'old enfeir cloth':6,'old zinzibar cloth':6,'luxurious cloth':8},
             'Quests':{'cooling cubes'}}
game_launched = [False]
hovering = [0]

def make_positions(start_y=0):
    pos = {}
    for i in range(start_y, ytiles):
        for j in range(xtiles):
            tiletype = input(f'({j},{i}): ')
            if tiletype != '':
                if tiletype not in pos:
                    pos[tiletype] = [(j, i)]
                else:
                    pos[tiletype].append((j, i))
    return pos

def merge_positions(pos1, pos2):
    pos = {}
    for key in pos1:
        if key not in pos2:
            pos[key] = pos1[key]
        else:
            seen_tup = set()
            pos[key] = []
            for tup in pos1[key]:
                pos[key].append(tup)
                seen_tup.add(tup)
            for tup in pos2[key]:
                if tup not in seen_tup:
                    pos[key].append(tup)
    for key in pos2:
        if key not in pos:
            pos[key] = pos2[key]
    return pos

positions = {'road': [(3, 1),(4, 1),(5, 1),(6, 1),(8, 1),(9, 1),(10, 1),(11, 2),(12, 2),(13, 2),(2, 2),(3, 2),(3, 3),(1, 3),(1, 4),(1, 5),(2, 5),(3, 5),
                      (4, 6),(8, 2),(7, 3),(14, 3),(5, 4),(6, 4),(7, 4),(9, 4),(10, 4),(11, 4),(12, 4),(15, 4),(6, 5),(10, 5),(12, 5),(14, 5),(6, 6),(11, 6),
                      (13, 6),(14, 6),(1, 7),(5, 7),(6, 7),(7, 7),(8, 7),(9, 7),(10, 7),(12, 7),(2, 8),(4, 8),(12, 8),(2, 9),(3, 9),(4, 9),(12, 9),(5, 10),
                      (13, 10),(5, 11),(13, 11),(6, 12),(13, 12)],
             'randoms': [(4, 2),(5, 2),(6, 2),(2, 3),(9, 2),(10, 2),(11, 1),(5, 3),(6, 3),(9, 3),(13, 3),(2, 4),(13, 4),(14, 4),(5, 5),(9, 5),(11, 5),(13, 5),
                         (3, 6),(7, 6),(8, 6),(9, 6),(10, 6),(2, 7),(13, 7),(1, 8),(3, 8),(6, 8),(7, 8),(8, 8),(9, 8),(10, 8),(13, 8),(14, 8),(15, 8),(5, 9),
                         (6, 9),(7, 9),(8, 9),(9, 9),(10, 9),(11, 9),(13, 9),(14, 9),(4, 10),(6, 10),(11, 10),(12, 10),(14, 10),(3, 11),(4, 11),(6, 11),
                         (11, 11),(12, 11),(4, 12),(7, 12),(12, 12)],
             'ruins': [(1, 1), (2, 0), (3, 0), (14, 0), (15, 1)],
             'battle1': [(14, 2)], 'battle2': [(14, 1)],
             'village1': [(0, 5),(7, 0),(4, 3),(12, 1),(8, 3),(10, 3),(15, 3),(5, 6),(12, 6),(14, 7),(1, 9),(5, 12),(12, 13)],
             'village2': [(8, 0),(0, 6),(16, 2),(12, 3),(3, 4),(7, 5),(3, 7),(15, 7),(11, 8),(1, 10),(4, 13),(13, 14)],
             'village3': [(7, 2),(4, 5),(8, 5),(16, 6),(0, 7),(5, 8),(1, 11),(5, 14),(14, 14)],
             'village4': [(15, 5), (2, 6), (2, 11), (14, 13), (6, 14)],
             'village5': [(3, 10), (14, 12), (6, 13)],
             'fodker': [(2, 1)],'kubani': [(7, 1)],'scetcher': [(4, 4)],'pafiz': [(1, 6)],'tamariza': [(4, 7)],'zinzibar': [(15, 2)],'enfeir': [(13, 1)],
             'tamarania': [(11, 3)],'demetry': [(8, 4)],'starfex': [(15, 6)],'anafola': [(11, 7)],'tutalu': [(2, 10)],'benfriege': [(5, 13)],'glaser': [(13, 13)],
             'wilderness': [(7, 10),(8, 10),(9, 10),(10, 10),(7, 11),(8, 11),(9, 11),(10, 11),(8, 12),(9, 12),(10, 12),(11, 12),(7, 13),(11, 13)]}
randoms = ['oldlibrary','outpost','cave','mountain','plains','pond']

def get_dim(dx, dy):
    return int(xpix*dx), int((ypix * (dy-(dy//2))) + (xpix * (dy//2) / np.sqrt(3)))

xsize, ysize = get_dim(xtiles, ytiles)
scale = 1
xprel, yprel = scale * (xpix / xsize), scale * (ypix / ysize)

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

def get_hexcolor(rgb):
    return '%02x%02x%02x' % rgb

class Tile(ButtonBehavior, HoverBehavior, Image):
    def __init__(self, tile, x, y, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.source = f'images\\tile\\{tile}.png'
        self.hoveringOver = False
        self.parentBoard = None
        self.empty_label = None
        self.empty_label_rounds = None
        self.is_empty = False
        self.trader_label = None
        self.trader_rounds = 0
        self.trader_wares = set()
        self.trader_wares_label = None
        self.city_wares = set()
        self.neighbors = set()
        self.bind(on_press=self.initiate)
        self.tile = tile
        self.gridx = x
        self.gridy = y
        self.updateView()
    def empty_tile(self, rounds=6, recvd=False):
        if not recvd:
            socket_client.send('[EMPTY]',(self.gridx, self.gridy))
        self.opacity = 0.4
        self.empty_label_rounds=rounds
        self.empty_label = Label(text=str(rounds), bold=True,pos_hint=self.pos_hint, size_hint=self.size_hint, color=(0,0,0,1), markup=True, font_size=20)
        self.parentBoard.add_widget(self.empty_label)
        self.is_empty = True
    def trader_appears(self, rounds=3, recvd=False):
        if not recvd:
            Categories = list(gameItems)
            Categories.remove('Quests')
            Categories.remove('Cloth')
            randomCategories = np.random.choice(Categories, 4)
            self.trader_wares = set()
            for c in randomCategories:
                self.trader_wares.add(np.random.choice(list(gameItems[c])))
            socket_client.send('[TRADER]',[(self.gridx, self.gridy), self.trader_wares])
        else:
            self.trader_wares = recvd
        self.color = (0.5, 1, 0.5, 1)
        self.trader_label = Label(text=str(rounds), bold=True, pos_hint=self.pos_hint, size_hint=self.size_hint, color=(0,0,0,1), markup=True, font_size=20)
        self.trader_rounds = rounds
        self.parentBoard.add_widget(self.trader_label)
        self.parentBoard.sendFaceMessage('Trader Appears!','green')
    def buy_from_trader(self, item, rounds=3, recvd=False):
        if not recvd:
            socket_client.send('[TRADER]',[(self.gridx, self.gridy), item])
        self.trader_rounds = rounds
        self.trader_label.text = str(rounds)
        self.trader_wares.remove(item)
    def update_tile_properties(self):
        if self.is_empty:
            self.empty_label_rounds -= 1
            if self.empty_label_rounds == 0:
                self.empty_label_rounds = None
                self.empty_label.text = ''
                self.parentBoard.remove_widget(self.empty_label)
                self.empty_label = None
                self.is_empty = False
                self.opacity = 1
            else:
                self.empty_label.text = str(self.empty_label_rounds)
        if self.trader_rounds > 0:
            self.trader_rounds -= 1
            if self.trader_rounds == 0:
                self.trader_label.text = ''
                self.parentBoard.remove_widget(self.trader_label)
                self.trader_label = None
                self.color = (1, 1, 1, 1)
                self.trader_wares = set()
            else:
                self.trader_label.text = str(self.trader_rounds)
    def updateView(self, xshift=0, yshift=0):
        xpos, ypos = get_relpos(self.gridx, self.gridy)
        mag_x = 1 if self.parentBoard is None else self.parentBoard.zoom + 1
        mag_y = 1 if self.parentBoard is None else self.parentBoard.zoom + 1
        self.size_hint = (xprel * mag_x, yprel * mag_y)
        xpos, ypos = xpos * mag_x, ypos * mag_x
        self.pos_hint = {'x': xpos + xshift, 'y': ypos + yshift}
        if self.empty_label is not None:
            self.empty_label.pos_hint = self.pos_hint
            self.empty_label.size_hint = self.size_hint
        if self.trader_label is not None:
            self.trader_label.pos_hint = self.pos_hint
            self.trader_label.size_hint = self.size_hint
        self.centx, self.centy = xpos + xshift + (xprel*mag_x/2), ypos + yshift + (yprel*mag_y/2)
    def set_neighbors(self):
        neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1 if self.gridy % 2 else -1, 1], [1 if self.gridy % 2 else -1, -1]]
        self.neighbors, self.neighbortiles = set(), set()
        for dx, dy in neighbors:
            x, y = self.gridx + dx, self.gridy + dy
            if (x, y) in self.parentBoard.gridtiles:
                self.neighbors.add((x, y))
                self.neighbortiles.add(self.parentBoard.gridtiles[(x,y)].tile)
    def is_neighbor(self):
        return self.parentBoard.localPlayer.currentcoord in self.neighbors
    def findNearest(self, tiles, T=None, depth=0, checked=set(), queue={}):
        if T is None: T = self
        if T.tile in tiles:
            nextDepth = None
            return (T.gridx, T.gridy), depth
        nextinLine = deepcopy(T.neighbors.difference(checked))
        checked = checked.union(nextinLine)
        if (depth+1) in queue:
            queue[depth+1] = queue[depth+1].union(nextinLine)
        else:
            queue[depth+1] = nextinLine
        nextDepth = min(queue.keys())
        if len(queue[nextDepth]) == 0:
            queue.pop(nextDepth)
            nextDepth = min(queue.keys())
            if len(queue[nextDepth]) == 0:
                print("Warning: are you sure you are looking for a tile that exists?")
        nextT = self.parentBoard.gridtiles[queue[nextDepth].pop()]
        return nextT.findNearest(tiles, nextT, nextDepth, checked, queue)
    def on_enter(self, *args):
        hovering[0] += 1
        self.source = f'images\\selectedtile\\{self.tile}.png'
        if (self.trader_rounds > 0) and (self.trader_wares_label is None):
            L = []
            for item in self.trader_wares:
                for categ in gameItems:
                    if item in gameItems[categ]:
                        L.append(f'{item}: [color=ffff75]{gameItems[categ][item]}[/color]')
                        break
            self.trader_wares_label = Button(text='\n'.join(L), color=(1, 1, 1, 1), markup=True, pos=(self.pos[0]+self.size[0]/2,self.pos[1]+self.size[1]/2), size_hint=(self.size_hint[0]*4/(self.parentBoard.zoom+1),self.size_hint[1]*1.5/(self.parentBoard.zoom+1)), background_color=(0.3, 0.3, 0.3, 0.7))
            self.parentBoard.add_widget(self.trader_wares_label)
    def on_leave(self, *args):
        hovering[0] -= 1
        self.source = f'images\\tile\\{self.tile}.png'
        if self.trader_wares_label is not None:
            self.trader_wares_label.text = ''
            self.parentBoard.remove_widget(self.trader_wares_label)
            self.trader_wares_label = None
    def initiate(self, instance):
        if self.parentBoard.localPlayer.paused:
            return
        elif hovering[0] > 1:
            output("Hovering over too many tiles! No action performed!",'yellow')
            return
        # If the tile is a neighbor and the player is not descended in a cave or ontop of a mountain... (tiered)
        elif self.is_neighbor() and (not self.parentBoard.localPlayer.tiered):
            self.parentBoard.localPlayer.moveto((self.gridx, self.gridy))

trns_clr = {'red':get_hexcolor((255,0,0)),'green':get_hexcolor((0,255,0)),'yellow':get_hexcolor((147,136,21)),'blue':get_hexcolor((0,0,255))}
class BoardPage(FloatLayout):
    def __init__(self, game_page, **kwargs):
        super().__init__(**kwargs)
        self.game_page = game_page
        self.localuser = game_app.connect_page.username.text
        self.size = get_dim(xtiles, ytiles)
        self.zoom = 0
        self.gridtiles = {}
        self.Players = {}
        self.localPlayer = None
        self.startLblMsgs = []
        self.startLblTimes = []
        for tiletype in positions:
            if tiletype == 'randoms':
                continue
            for x, y in positions[tiletype]: self.add_tile(tiletype, x, y)
        np.random.seed(seed)
        randomChoice = np.random.choice(randoms*int(np.ceil(len(positions['randoms'])/len(randoms))), len(positions['randoms']))
        for i in range(len(positions['randoms'])):
            x, y = positions['randoms'][i]
            self.add_tile(randomChoice[i], x, y)
        for T in self.gridtiles.values():
            T.set_neighbors()
        self.zoomButton = Button(text="Zoom", pos_hint={'x':0,'y':0}, size_hint=(0.06, 0.03))
        self.zoomButton.bind(on_press=self.updateView)
        self.add_widget(self.zoomButton)
    def add_tile(self, tiletype, x, y):
        T = Tile(tiletype, x, y)
        self.gridtiles[(x,y)] = T
        T.parentBoard = self
        self.add_widget(T)
    def add_player(self, username, birthcity):
        self.Players[username] = Player(self, username, birthcity)
        self.add_widget(self.Players[username])
        if username == self.localuser:
            # add local player
            self.localPlayer = self.Players[username]
            # So that the start round will appear above the player
            self.startLabel = Label(text='',bold=True,color=(0.5, 0, 1, 0.7),pos_hint={'x':0,'y':0},size_hint=(1,1),font_size=50,markup=True)
            self.add_widget(self.startLabel)
            self.game_page.recover_button.bind(on_press=self.localPlayer.recover)
            self.game_page.eat_button.bind(on_press=self.localPlayer.eat())
            self.localPlayer.add_mainStatPage()
            self.localPlayer.add_PlayerTrack()
    def sendFaceMessage(self, msg=None, clr=None, scheduleTime=2):
        timeNow = time()
        msg = f'[color={trns_clr[clr]}]{msg}[/color]' if clr is not None else msg
        if clr is not None:
            self.startLblMsgs.append(msg)
            self.startLblTimes.append(timeNow)
        max_i = -1
        for i in range(len(self.startLblTimes)):
            if (timeNow - self.startLblTimes[i]) > scheduleTime:
                max_i = i
        self.startLblMsgs = self.startLblMsgs[(max_i+1):]
        self.startLblTimes = self.startLblTimes[(max_i+1):]
        self.startLabel.text = '\n'.join(self.startLblMsgs)
        if msg is not None: output(msg, clr)
        def clear_lbl(_):
            self.sendFaceMessage()
        Clock.schedule_once(clear_lbl, scheduleTime)
    def updateView(self, deltaZoom=0):
        # Make sure change does not go beyond the bounds of [0,3]
        if deltaZoom:
            self.zoom = 0 if self.zoom == 3 else self.zoom+1
        if (self.zoom == 0) and (type(deltaZoom) is int) and (deltaZoom==0):
            # We still update the players because of initiation troubles
            for P in self.Players.values():
                P.updateView()
            return
        lrpx, lrpy = get_relpos(self.localPlayer.currenttile.gridx, self.localPlayer.currenttile.gridy)
        lrpx *= (self.zoom+1)
        lrpy *= (self.zoom+1)
        if self.zoom > 0:
            shiftx, shifty = (0.5 - (xprel*(self.zoom+1)/2) - lrpx), (0.5 - (yprel*(self.zoom+1)/2) - lrpy)
        else:
            shiftx, shifty = 0, 0
        for T in self.gridtiles.values():
            T.updateView(shiftx, shifty)
        for P in self.Players.values():
            P.updateView()
            
class GamePage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2
        self.button_num = 1
        # Add the Main Screen
        self.main_screen = ScreenManager()
        self.board_page = BoardPage(self)
        screen = Screen(name='Board')
        screen.add_widget(self.board_page)
        self.main_screen.add_widget(screen)
        self.add_widget(self.main_screen)
        # Add the Side Screen
        input_y, label_y, stat_y, action_y, output_y, toggle_y = 0.05, 0.25, 0.1, 0.2, 0.35, 0.05
        self.stat_ypos, self.stat_ysize = input_y+label_y, stat_y
        self.right_line = RelativeLayout(size_hint_x=0.25)
        self.toggleView = Button(text="Player Track",pos_hint={'x':0,'y':(input_y+label_y+stat_y+action_y+output_y)},size_hint=(1,toggle_y),background_color=(0, 0.4, 0.4, 1))
        self.toggleView.bind(on_press=self.switchView)
        self.output = Label(text='Action Buttons:',pos_hint={'x':0,'y':(input_y+label_y+stat_y+action_y)},size_hint=(1,output_y),color=(0.1,0.1,0.1,0.8),valign='bottom',halign='left',markup=True)
        self.output.text_size = self.output.size
        with self.output.canvas.before:
            Color(0.6, 0.6, 0.8, 0.7, mode='rgba')
            self.outrect=Rectangle(pos=self.output.pos, size=self.output.size)
        self.output.bind(pos=self.update_bkgSize, size=self.update_bkgSize)
        self.actionGrid = GridLayout(pos_hint={'x':0,'y':(input_y+label_y+stat_y)},size_hint_y=action_y,cols=2)
        self.recover_button = Button(text='Rest (2)')
        self.eat_button = Button(text='Eat')
        # Button is bound after local player is detected
        self.actionGrid.add_widget(self.recover_button)
        self.actionGrid.add_widget(self.eat_button)
        self.actionButtons = [self.recover_button, self.eat_button]
        self.statGrid = GridLayout(pos_hint={'x':0,'y':(input_y+label_y)},size_hint_y=stat_y,cols=4)
        self.display_page = Label(text='Chat Box (press enter to send message)\n',pos_hint={'x':0,'y':input_y},color=(1,1,1,0.8),size_hint=(1,label_y),markup=True)
        self.display_page.text_size = self.display_page.size
        with self.display_page.canvas.before:
            Color(0, 0, 0, 0.7, mode='rgba')
            self.rect=Rectangle(pos=self.display_page.pos,size=self.display_page.size)
        # Make the display background update itself evertime there is a window size change
        self.display_page.bind(pos=self.update_bkgSize,size=self.update_bkgSize)
        self.new_message = TextInput(pos_hint={'x':0,'y':0},size_hint=(1,input_y))
        # Append the widgets
        self.right_line.add_widget(self.toggleView)
        self.right_line.add_widget(self.output)
        self.right_line.add_widget(self.actionGrid)
        self.right_line.add_widget(self.display_page)
        self.right_line.add_widget(self.new_message)
        self.add_widget(self.right_line)
        # Any keyboard press will trigger the event:
        Window.bind(on_key_down=self.on_key_down)
    def switchView(self, instance):
        self.main_screen.current = self.toggleView.text
        self.toggleView.text = "Board" if self.toggleView.text=="Player Track" else "Player Track"
    def make_actionGrid(self, funcDict, save_rest=False):
        self.clear_actionGrid(save_rest)
        for txt, func in funcDict.items():
            B = Button(text=txt, markup=True)
            B.bind(on_press=func)
            self.actionButtons.append(B)
            self.actionGrid.add_widget(B)
    def clear_actionGrid(self, save_rest=False):
        for B in self.actionButtons:
            self.actionGrid.remove_widget(B)
        if save_rest:
            self.actionButtons = [self.recover_button, self.eat_button]
            self.actionGrid.add_widget(self.recover_button)
            self.actionGrid.add_widget(self.eat_button)
        else:
            self.actionButtons = []
    def update_bkgSize(self, instance, value):
        self.rect.size = self.display_page.size
        self.rect.pos = self.display_page.pos
        self.outrect.size = self.output.size
        self.outrect.pos = self.output.pos
        self.display_page.text_size = self.display_page.size
        self.output.text_size = self.output.size
    def update_output(self, message, color=None):
        if color is None:
            message = message
        elif color in trns_clr:
            message = '[color='+trns_clr[color]+']'+message+'[/color]'
        else:
            hx = get_hexcolor(color)
            message = '[color='+hx+']'+message+'[/color]'
        self.output.text += '\n' + message
    def update_display(self, username, message):
        if message[0]=='\n':
            message = message[1:]
        clr = get_hexcolor((131,215,190)) if username == self.board_page.localuser else get_hexcolor((211, 131, 131))
        self.display_page.text += f'\n|[color={clr}]{username}[/color]| ' + message
    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        # We want to take an action only when Enter key is being pressed, and send a message
        if keycode == 40:
            # Send Message
            message = self.new_message.text
            self.new_message.text = ''
            if message:
                self.update_display(self.board_page.localuser, message)
                socket_client.send('[CHAT]',message)
        
class BirthCityButton(ButtonBehavior, HoverBehavior, Image):
    def __init__(self, parentPage, city, **kwargs):
        super(BirthCityButton, self).__init__(**kwargs)
        self.parentPage = parentPage
        self.city = city
        self.claimed = False
        self.claim_lbl = Label(halign='center',valign='middle',markup=True)
        self.source = f'images\\tile\\{self.city}.png'
        self.bind(on_press=self.go_to_board)
    def on_enter(self, *args):
        self.source = f'images\\selectedtile\\{self.city}.png'
        benefits = cities[self.city]
        ks, cb = benefits['Knowledges'], benefits['Combat Boosts']
        kstr = ', '.join([(str(ks[i][1])+' ' if ks[i][1]>1 else '')+ks[i][0] for i in range(len(ks))]) if len(ks)>0 else '-'
        cstr = ', '.join([(str(cb[i][1])+' ' if cb[i][1]>1 else '')+cb[i][0] for i in range(len(cb))]) if len(cb)>0 else '-'
        display = '[b][color=ffa500]'+self.city[0].upper()+self.city[1:]+'[/color][/b]:\nCombat Style: [color=ffa500]'+benefits['Combat Style']+'[/color]\nCoins: [color=ffa500]'+str(benefits['Coins'])
        display += '[/color]\nKnowledges: [color=ffa500]'+kstr+'[/color]\nCombat Boosts: [color=ffa500]'+cstr
        self.parentPage.displaylbl.text = '[color=000000]'+display+'[/color][/color]'
        self.parentPage.character_image.source = f'images\\characters\\{self.city}.png'
    def on_leave(self, *args):
        self.source = f'images\\tile\\{self.city}.png'
    def make_claim(self):
        self.claimed = True
        self.claim_lbl.text = '[color=ff0000][b]CLAIMED[/b][/color]'
    def go_to_board(self, instance):
        if not self.claimed:
            socket_client.send("[CLAIM]",self.city)
            self.parentPage.make_claim(game_app.launch_page.username, self.city)
            game_app.screen_manager.current = 'Game'
        
class BirthCityPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols=7
        self.cityorder = sorted(cities.keys())
        self.bcbs = {}
        for city in self.cityorder:
            self.bcbs[city] = BirthCityButton(self, city=city)
            mixed_cell = RelativeLayout()
            mixed_cell.add_widget(self.bcbs[city])
            mixed_cell.add_widget(self.bcbs[city].claim_lbl)
            self.add_widget(mixed_cell)
        self.character_image = Image()
        self.add_widget(self.character_image)
        self.add_widget(Label())
        self.add_widget(Label())
        self.displaylbl = Label(text='[color=000000]Hover over a city to see their benefits![/color]', markup=True,
                                halign='center',valign='middle')
        self.add_widget(self.displaylbl)
    def make_claim(self, username, city):
        self.bcbs[city].make_claim()
        game_app.game_page.board_page.add_player(username, city)
        
class LaunchPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1
        self.username = game_app.connect_page.username.text
        self.usernames, self.ready = [self.username], {self.username:0}
        self.label = Label(text=f'[color=000000]{self.username}: Not Ready[/color]', height=Window.size[1]*0.9, size_hint_y=None, markup=True)
        self.add_widget(self.label)
        self.readyButton = Button(text="Ready Up")
        self.readyButton.bind(on_press=self.update_self)
        self.add_widget(self.readyButton)
        socket_client.start_listening(self.incoming_message, show_error)
        socket_client.send("[LAUNCH]","Listening")
    def check_launch(self):
        all_ready = True
        for value in self.ready.values():
            if value == 0:
                all_ready = False
                break
        if all_ready:
            game_launched[0] = True
            game_app.screen_manager.current = 'Birth City Chooser'
    def refresh_label(self):
        self.label.text = '[color=000000]'+'\n'.join([self.usernames[i]+': '+['Not Ready', 'Ready'][self.ready[self.usernames[i]]] for i in range(len(self.usernames))])+'[/color]'
        self.check_launch()
    def incoming_message(self, username, category, message):
        if category == '[LAUNCH]':
            if username not in self.ready:
                self.usernames.append(username)
            self.ready[username] = 1 if message == 'Ready' else 0
            self.refresh_label()
        elif (category == '[CONNECTION]') and (message == 'Closed'):
            self.usernames.remove(username)
            self.ready.pop(username)
            print(f'{username} Lost Connection!')
            output(f'{username} Lost Connection!')
            if game_launched[0] == False:
                self.refresh_label()
        elif category == '[CLAIM]':
            output(f"{username} claimed {message}")
            def clockedClaim(_):
                game_app.chooseCity_page.make_claim(username, message)
            #game_app.chooseCity_page.make_claim(username, message)
            Clock.schedule_once(clockedClaim, 0.1)
        elif category == '[MOVE]':
            def clockedMove(_):
                game_app.game_page.board_page.Players[username].moveto(message, False, True)
            Clock.schedule_once(clockedMove, 0.2)
        elif category == '[CHAT]':
            def clockedChat(_):
                game_app.game_page.update_display(username, message)
            Clock.schedule_once(clockedChat, 0.2)
        elif category == '[EMPTY]':
            def emptyTile(_):
                game_app.game_page.board_page.gridtiles[message].empty_tile(recvd=True)
            Clock.schedule_once(emptyTile, 0.2)
        elif (category == '[ROUND]') and (message == 'end'):
            def pauseUser(_):
                game_app.game_page.board_page.Players[username].pause()
            Clock.schedule_once(pauseUser, 0.2)
        elif (category == '[TRADER]'):
            def clockedTrader(_):
                game_app.game_page.board_page.gridtiles[message[0]].trader_appears(recvd=message[1])
            def clockedPurchase(_):
                game_app.game_page.board_page.gridtiles[message[0]].buy_from_trader(message[1], recvd=True)
            if type(message[1]) is set:
                Clock.schedule_once(clockedTrader, 0.2)
            elif type(message[1]) is str:
                Clock.schedule_once(clockedPurchase, 0.1)
    def update_self(self, _):
        self.ready[self.username] = 1 - self.ready[self.username]
        # send the message to the server that they are ready
        if self.ready[self.username]:
            socket_client.send("[LAUNCH]","Ready")
            self.readyButton.text = "Not Ready Anymore" 
        else:
            socket_client.send("[LAUNCH]","Not Ready")
            self.readyButton.text = "Ready Up"
        self.refresh_label()
            
        
# Simple information/error page
class InfoPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Just one column
        self.cols = 1

        # And one label with bigger font and centered text
        self.message = Label(halign="center", valign="middle", font_size=30)

        # By default every widget returns it's side as [100, 100], it gets finally resized,
        # but we have to listen for size change to get a new one
        # more: https://github.com/kivy/kivy/issues/1044
        self.message.bind(width=self.update_text_width)

        # Add text widget to the layout
        self.add_widget(self.message)

    # Called with a message, to update message text in widget
    def update_info(self, message):
        self.message.text = message

    # Called on label width update, so we can set text width properly - to 90% of label width
    def update_text_width(self, *_):
        self.message.text_size = (self.message.width * 0.9, None)
        
class ConnectPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 2  # used for our grid
        if os.path.isfile("prev_details.txt"):
            with open("prev_details.txt","r") as f:
                d = f.read().split(",")
                prev_ip = d[0]
                prev_port = d[1]
                prev_username = d[2]
        else:
            prev_ip, prev_port, prev_username = '','',''

        self.add_widget(Label(text='IP:'))  # widget #1, top left
        self.ip = TextInput(text=prev_ip, multiline=False)  # defining self.ip...
        self.add_widget(self.ip) # widget #2, top right

        self.add_widget(Label(text='Port:'))
        self.port = TextInput(text=prev_port, multiline=False)
        self.add_widget(self.port)

        self.add_widget(Label(text='Username:'))
        self.username = TextInput(text=prev_username, multiline=False)
        self.add_widget(self.username)

        # add our button.
        self.join = Button(text="Join")
        self.join.bind(on_press=self.join_button)
        self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.join)

    def join_button(self, instance):
        port = self.port.text
        ip = self.ip.text
        username = self.username.text
        with open("prev_details.txt","w") as f:
            f.write(f"{ip},{port},{username}")
        # Create info string, update InfoPage with a message and show it
        info = f"Joining {ip}:{port} as {username}"
        game_app.info_page.update_info(info)
        game_app.screen_manager.current = 'Info'
        Clock.schedule_once(self.connect, 1)

    # Connects to the server
    # (second parameter is the time after which this function had been called,
    #  we don't care about it, but kivy sends it, so we have to receive it)
    def connect(self, _):

        # Get information for sockets client
        port = int(self.port.text)
        ip = self.ip.text
        username = self.username.text

        if not socket_client.connect(ip, port, username, show_error):
            return

        # Create chat page and activate it
        game_app.start_game_screen()
        game_app.screen_manager.current = 'Launcher'
        

        
class EpicApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        
        # Connect Page
        self.connect_page = ConnectPage()
        screen = Screen(name='Connect')
        screen.add_widget(self.connect_page)
        self.screen_manager.add_widget(screen)
        
        # Info page
        self.info_page = InfoPage()
        screen = Screen(name='Info')
        screen.add_widget(self.info_page)
        self.screen_manager.add_widget(screen)
        return self.screen_manager
        
    def start_game_screen(self):
        Window.clearcolor = (1, 1, 1, 1)
        # Launch Page
        self.launch_page = LaunchPage()
        screen = Screen(name = 'Launcher')
        screen.add_widget(self.launch_page)
        self.screen_manager.add_widget(screen)
        
        # City chooser
        self.chooseCity_page = BirthCityPage()
        screen = Screen(name='Birth City Chooser')
        screen.add_widget(self.chooseCity_page)
        self.screen_manager.add_widget(screen)
        
        # The Game Page
        self.game_page = GamePage()
        screen = Screen(name='Game')
        screen.add_widget(self.game_page)
        self.screen_manager.add_widget(screen)
        
#        # The board
#        self.board_page = BoardPage()
#        screen = Screen(name='Board')
#        screen.add_widget(self.board_page)
#        self.screen_manager.add_widget(screen)
        

# Error callback function, used by sockets client
# Updates info page with an error message, shows message and schedules exit in 10 seconds
# time.sleep() won't work here - will block Kivy and page with error message won't show up
def show_error(message):
    game_app.info_page.update_info(message)
    game_app.screen_manager.current = 'Info'
    Clock.schedule_once(sys.exit, 10)

if __name__ == "__main__":
    game_app = EpicApp()
    game_app.run()