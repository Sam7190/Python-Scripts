"""
cd Documents\\GitHub\\Python-Scripts\\elementalSword
python game.py

# Installations:
conda install kivy -c conda-forge # Install kivy - make sure it is v1.11.1
pip install kivymd # Install kivymd - make sure it is v0.104.1
This was using python version 7.3.7 and pip version 21.0.1

# Specific version installation:
conda install kivy==1.11.1 -c conda-forge
pip install kivymd==0.104.1
"""
#%% Import Modules

# Import helper modules
import buildAI as AI
#import skillGames
import hallmarks as hmrk
import gameVariables as var
#import fightingSystem as fightsys
import essentialfuncs as essf
import person_quest_positions as pqp
from common_widgets import LockIcon, SkirmishIcon, HoverButton, Table, HoveringLabel, ScrollLabel, ActionButton

# Import standard modules
import numpy as np
import pandas as pd
from PIL import Image as pilImage
from skimage import color as skcolor
import matplotlib.path as mplPath
import os
import sys
import csv
import pickle
import logging
from inspect import currentframe, getframeinfo
from collections import Counter
from functools import partial # This would have made a lot of nested functions unnecessary! (if I had known about it earlier)
from copy import deepcopy
import socket_client
import socket_server
from time import time
from win32api import GetSystemMetrics, GetMonitorInfo, MonitorFromPoint
#import mechanic as mcnc

# Import kivy modules
import kivy
from kivy.app import App
from kivy.uix.dropdown import DropDown
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
from kivy.graphics import Color,Rectangle,Ellipse,InstructionGroup
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty
from kivy.clock import Clock
from kivy.core.window import Window
from PIL import Image as PILImage

#%% Settings and Core Functions

# Settings
npcDifficulty = 3
seed = None
auto_resize = True
save_file = None
load_file = 'None'
gameEnd = {2: 100}

# Tile Resolution and number of tiles in gameboard.
xpix = 343
ypix = 396
xtiles = 17
ytiles = 15

def lclPlayer():
    return game_app.game_page.board_page.localPlayer
def rbtwn(mn, mx, size=None, chance=None, msg=''):
    r = np.random.choice(np.arange(mn, mx+1), size)
    if (chance is not None):
        if chance < mn:
            chance = 0
        elif chance > mx:
            chance = 1
        else:
            chance = chance - mn + 1
        output(f'{msg}Success chance is {chance} in {mx-mn+1}')
    return r
def Clocked(function, delay, debug_msg=None):
    if debug_msg is not None: logging.debug(f'Clocked @{delay}sec | {debug_msg}')
    Clock.schedule_once(function, delay)

#%% Fighting System

class HitBox(Button):
    def __init__(self, fpage, **kwargs):
        super().__init__(**kwargs)
        logging.debug("Making hitbox")
        self.fpage = fpage
        self.xposScale = 1 + self.fpage.P.parentBoard.game_page.right_line_x
        self.timeLapseRestrained = 1.5 - 0.1*self.fpage.foestats[self.fpage.P.attributes['Agility']]
        self.timeLapseAllowed = self.fpage.pstats[self.fpage.P.attributes["Agility"]] + 1#0.5 + 0.25 * self.fpage.pstats[self.fpage.P.attributes["Agility"]]
        self.technique = self.fpage.pstats[self.fpage.P.attributes['Technique']]
        self.fakesRemaining = self.fpage.pstats[self.fpage.P.attributes['Cunning']]
        self.appliedTechnique = 0
        # The next four lines are defined again in transformBoxes function of FightPage!
        self.boxMaxScale = self.fpage.size[0]*0.04 # The approximate game_board_x times an arbitrary max ratio
        self.bxAdj = self.boxMaxScale/2
        self.techniqueScale = self.fpage.size[0]*0.00304
        self.boxMinScale = self.boxMaxScale - self.techniqueScale*self.technique
        self.boxes = []
        self.fakes = []
        self.clrs = []
        self.cleared = []
        #self.lbls = []
        self.max_boxes = self.fpage.curAtk
        self.TimeRemaining = 1.5 + np.mean([self.max_boxes, self.timeLapseAllowed]) * self.timeLapseRestrained #self.max_boxes # Add a 1.5 second buffer just in case.
        self.TimeRemainingIter = 0.1
        self.curBox = None
        self.minBox = None
        self.attackEnded = False
        self.nextisFake = False
        self.TimeRemainingDisplay = Button(text=f"Time Remaining:\n{np.round(self.TimeRemaining,1)} seconds", pos_hint={'right':1, 'top':1}, size_hint=(0.2, 0.1), disabled=True, background_color=(1, 1, 3, 20))
        self.fpage.add_widget(self.TimeRemainingDisplay)
        Window.bind(on_key_down=self.on_key_down)
        Clocked(self.endAttack, self.TimeRemaining, 'End Attack')
        Clock.schedule_interval(self.countDown, self.TimeRemainingIter)
    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        if (text == 'f') and (len(self.fakes) > 0) and (not self.fakes[-1]) and (self.fakesRemaining > 0):
            self.fakes[-1] = True
            self.clrs[-1].rgba = (0.9, 0.9, 0.9, 0.6)
            self.fakesRemaining -= 1
        elif (text == 'f') and (not self.attackEnded) and (self.fakesRemaining > 0):
            self.nextisFake = True
    def on_touch_down(self, touch):
        if ((len(self.boxes) - np.sum(self.fakes)) < self.max_boxes) and (not self.attackEnded):
            self.curxs = [touch.pos[0] - self.bxAdj, touch.pos[0] + self.bxAdj]
            self.curys = [touch.pos[1] - self.bxAdj, touch.pos[1] + self.bxAdj]
            if (self.curxs[0] >= self.pos[0]) and (self.curxs[1] <= (self.pos[0]+self.size[0])) and (self.curys[0] >= self.pos[1]) and (self.curys[1] <= (self.pos[1]+self.size[1])):
                self.touchpos = touch.pos
                size = self.boxMaxScale, self.boxMaxScale
                with self.fpage.canvas.after:
                    self.curBoxClr = Color(0.3, 0.3, 0.3, 1)
                    self.curBox = Rectangle(pos=(touch.pos[0]-size[0]/2, touch.pos[1]-size[1]/2), size=size)
                    self.minBoxClr = Color(0, 1, 0.2, 1)
                    self.minBox = Rectangle(pos=(touch.pos[0]-self.boxMinScale/2, touch.pos[1]-self.boxMinScale/2), size=(self.boxMinScale, self.boxMinScale))
                Clock.schedule_interval(self.applyTechnique, 0.1)
    def on_touch_move(self, touch):
        if (self.curBox is not None):# and (not self.curBox.disabled):
            self.curxs = [touch.pos[0] - self.curBox.size[0]/2, touch.pos[0] + self.curBox.size[0]/2]
            self.curys = [touch.pos[1] - self.curBox.size[1]/2, touch.pos[1] + self.curBox.size[1]/2]
            if (self.curxs[0] >= self.pos[0]) and (self.curxs[1] <= (self.pos[0]+self.size[0])) and (self.curys[0] >= self.pos[1]) and (self.curys[1] <= (self.pos[1]+self.size[1])):
                self.touchpos = touch.pos
                self.curBox.pos = (touch.pos[0] - self.curBox.size[0]/2, touch.pos[1] - self.curBox.size[1]/2)
                self.minBox.pos = (touch.pos[0] - self.boxMinScale/2, touch.pos[1] - self.boxMinScale/2)
    def addCurBox(self):
        if self.nextisFake:
            self.curBoxClr.rgba = (0.9, 0.9, 0.9, 0.6)
            self.fakes.append(True)
            self.nextisFake = False
            self.fakesRemaining -= 1
        else:
            self.curBoxClr.rgba = (0.8, 0, 0.8, 0.6)
            self.fakes.append(False)
        self.fpage.canvas.after.remove(self.minBox)
        self.boxes.append(self.curBox)
        self.clrs.append(self.curBoxClr)
        self.cleared.append(False)
        self.curBox = None
        self.minBox = None
        if (len(self.boxes) - np.sum(self.fakes)) >= self.max_boxes:
            self.endAttack()
    def on_touch_up(self, touch):
        if self.curBox is not None:
            self.addCurBox()
    def applyTechnique(self, instance=None):
        if self.curBox is not None:
            if self.appliedTechnique < self.technique:
                newsize = self.curBox.size[0] - self.techniqueScale, self.curBox.size[1] - self.techniqueScale
                self.appliedTechnique += 1
            else:
                newsize = self.boxMaxScale, self.boxMaxScale
                self.appliedTechnique = 0
            self.curBox.size = newsize
            self.curBox.pos = (self.touchpos[0] - newsize[0]/2, self.touchpos[1] - newsize[1]/2)
        else:
            return False
    def endAttack(self, instance=None):
        if not self.attackEnded:
            self.attackEnded = True
            if self.curBox is not None:
                self.addCurBox()
            self.fpage.remove_widget(self.TimeRemainingDisplay)
            Window.unbind(on_key_down=self.on_key_down)
            if self.fpage.foeisNPC:
                self.fpage.msgBoard.text = "NPC Defending"
                Clocked(self.npcDefends, 1, "NPC Defending")
            else:
                # [INCOMPLETE] Send attack data to the defending user
                pass
    def getNormalizedPosition(self, pos):
        xMin, xMax, yMin, yMax = self.fpage.size[0]*0.02, self.fpage.size[0]*0.32, self.fpage.size[1]*0.02, self.fpage.size[1]*0.72
        return np.array([(pos[0] - xMin)/(xMax - xMin), (pos[1] - yMin)/(yMax - yMin)])
    def npcDefends(self, _=None):
        blocksLeft = self.fpage.foestats[self.fpage.P.attributes[f"Def-{self.fpage.P.combatstyle}"]]
        foeAgility = self.fpage.pstats[self.fpage.P.attributes["Agility"]]
        cunning = self.fpage.foestats[self.fpage.P.attributes["Cunning"]]
        stability = self.fpage.foestats[self.fpage.P.attributes["Stability"]]
        dodging = self.fpage.foestats[self.fpage.P.attributes["Agility"]]
        group2boxes, boxes2group, volley, click_prob, dodge_prob = {}, [], 0, [], []
        for i in range(len(self.boxes)):
            boxes2group.append(volley)
            if volley in group2boxes:
                group2boxes[volley].append(i)
            else:
                group2boxes[volley] = [i]
            if not self.fakes[i]:
                volley += 1
        for i in range(len(self.boxes)):
            tApplied = int(-(self.boxes[i].size[0] - self.boxMaxScale)/self.techniqueScale)
            vgID = boxes2group[i]
            gSize = len(group2boxes[vgID])
            nrmPos = self.getNormalizedPosition(self.boxes[i].pos)
            dis2cent = essf.euc(nrmPos, np.array([0.5, 0.5]))
            if vgID > 0:
                for gi in group2boxes[vgID-1]:
                    if not self.fakes[gi]:
                        dis2last = essf.euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
                        break
            else:
                dis2last = dis2cent
            total, dis2group = 0, 0
            for gi in group2boxes[vgID]:
                if gi == i:
                    continue
                total += 1
                dis2group += essf.euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
            if total > 0: dis2group /= total
            opacity = 0.96 if self.fakes[i] else (0.92 - cunning*0.013)
            data = [[foeAgility, cunning, stability, dodging, vgID, gSize, tApplied, dis2last, dis2cent, dis2group, opacity]]
            # Allow for a 1% bias correction to the model to avoid 100% click rates, and allow for reduce click probability as a function of volley number and order count
            click_prob.append(AI.cmdl.predict_proba(data)[:,1][0]*0.99*(1 - min([0.8, (0.08 * vgID/len(group2boxes))*(self.fpage.order_refresh_count // 3)])))
            dodge_prob.append(AI.dmdl.predict_proba(data)[:,1][0]*0.99*(1 - min([0.8, (0.1 * vgID/len(group2boxes))*(self.fpage.order_refresh_count // 3)])))
            # Apply NPC Difficulty to Click probability
            if npcDifficulty < 3:
                click_prob[-1] *= (1 / (2.5 - (1 - npcDifficulty)))
                dodge_prob[-1] *= (1 / (2.5 - (1 - npcDifficulty)))
            else:
                click_prob[-1] **= (1 / (npcDifficulty-2))
                dodge_prob[-1] **= (1 / (npcDifficulty-2))
        for vgID, boxIDs in sorted(group2boxes.items()):
            clicked_fake = True
            while clicked_fake and (len(boxIDs)>0):
                click_ps = [click_prob[i] for i in boxIDs]
                # Choose the box the NPC is most likely to click
                click_i = np.argmax(click_ps)
                i = boxIDs.pop(click_i)
                click_p, dodge_p = click_prob[i], dodge_prob[i]
                logging.info(f"Click Probability {click_p} {self.fakes[i]}")
                clickType = None
                if np.random.rand() <= click_p:
                    if not self.fakes[i]: clicked_fake = False
                    # This means the AI "clicked" the box - now check to see if they would have "dodged" the box
                    logging.info(f"Clicked! Dodge Probability {dodge_p} {self.fakes[i]}")
                    if np.random.rand() <= dodge_p:
                        # AI dodges the box -- pass because nothing happens
                        clickType = 'dodge'
                    elif blocksLeft > 0:
                        # AI blocks the box
                        blocksLeft -= 1
                        clickType = 'block'
                    elif not self.fakes[i]:
                        # NPC takes damage because they have no blocks left, and the box is not a fake
                        clickType = 'hit'
                elif not self.fakes[i]:
                    # NPC takes damage because the box is not a fake
                    clickType = 'hit'
                logging.info(f"Click Type: {clickType}")
                self.npcClickBox(i, clickType, 0.3 + 0.3*vgID, 0.3)
            for i in boxIDs:
                # Any remaining ids must be fake, so remove them
                self.npcClickBox(i, None, 0.3 + 0.3*vgID, 0.3)
            #Clocked(partial(self.removeBox, i, damageNPC), 0.3 + 0.3*vgID)
        try:
            vgID
        except NameError:
            vgID = 0
        Clocked(self.fpage.nextAttack, 0.9 + 0.3*vgID, "Go to Next Attack")
    def countDown(self, instance=None):
        if self.attackEnded:
            return False
        self.TimeRemaining = max([0, self.TimeRemaining-self.TimeRemainingIter])
        self.TimeRemainingDisplay.text = f'Time Remaining:\n{np.round(self.TimeRemaining,1)} seconds'
    def npcClickBox(self, i, clickType, delay, rmvTime):
        def clk(_=None):
            dmg = False
            if clickType is None:
                self.clrs[i].rgba = (1, 1, 1, 1)
            elif clickType == 'dodge':
                self.clrs[i].rgba = (0, 1, 0, 1)
            elif clickType == 'block':
                self.clrs[i].rgba = (0, 0, 1, 1)
            elif clickType == 'hit':
                self.clrs[i].rgba = (1, 0, 0, 1)
                dmg = True
            Clocked(partial(self.removeBox, i, dmg), rmvTime, "Remove hit box")
        Clocked(clk, delay, "NPC click box delay")
    def removeBox(self, i, damageFoe=False, _=None):
        if not self.cleared[i]:
            self.fpage.canvas.after.remove(self.boxes[i])
            self.cleared[i] = True
        if damageFoe:
            self.fpage.foeTakesDamage()
    def clearAll(self):
        for i in range(len(self.boxes)):
            self.removeBox(i)
        #for B in self.lbls:
        #    self.fpage.remove_widget(B)
        self.fpage.remove_widget(self)

class DefBox(Button):
    def __init__(self, fpage, **kwargs):
        super().__init__(**kwargs)
        self.fpage = fpage
        # The following two lines are also defined in HitBox!
        self.boxMaxScale = self.fpage.size[0]*0.04
        self.techniqueScale = self.fpage.size[0]*0.00304
        self.blocksRemaining = 50 if self.fpage.logDef else self.fpage.pstats[self.fpage.P.attributes[f'Def-{self.fpage.foestyle}']]
        self.dodging = self.fpage.pstats[self.fpage.P.attributes['Agility']] // 2
        self.cunning = self.fpage.pstats[self.fpage.P.attributes['Cunning']]
        self.mxOpacity, self.dOpacity = 0.92, 0.013
        self.stability = self.fpage.pstats[self.fpage.P.attributes['Stability']]
        self.foeagility = self.fpage.foestats[self.fpage.P.attributes['Agility']]
        self.fpage.msgBoard.text = ''
        self.totalTime = 1.5 + 0.15 * self.cunning
        # Every 3 rounds the amount of time you have to defend decreases by 10 percent, min being 0.2
        self.totalTime *= max([0.2, (1 - 0.1 * (self.fpage.order_refresh_count // 3))])
        self.dodgeTime = self.totalTime * (0.25 + 0.025*self.dodging) #0.5 + 0.2 * self.dodging
        self.blockTime = self.totalTime - self.dodgeTime #2 + 0.15 * self.cunning
        self.initState = 'dodge' if self.dodgeTime > 0 else 'block'
        self.strikeDelay = max([0.2, 2.3 - 0.14 * self.foeagility])
        self.fpage.msgBoard.text += f'Dodge Time: {round(self.dodgeTime,2)}\nBlock Time: {round(self.blockTime,2)}\nStrike Time: {round(self.strikeDelay,2)}'
        self.addExampleDisplay()
    def addExampleDisplay(self):
        x, y, size, buffer = float(self.fpage.size[0]*0.15), float(self.fpage.size[1]*0.8), (self.boxMaxScale, self.boxMaxScale), self.boxMaxScale*1.1
        with self.fpage.canvas.after:
            Color(0.2, 0.2, 0.6, 0.96)
            self.realBlock = Rectangle(pos=(x, y), size=size)
            Color(0.2, 0.2, 0.6, (self.mxOpacity-self.dOpacity*self.cunning))
            self.fakeBlock = Rectangle(pos=(x+buffer, y), size=size)
            Color(0.1, 0.8, 0.1, 0.96)
            self.realDodge = Rectangle(pos=(x, y+buffer), size=size)
            Color(0.1, 0.8, 0.1, (self.mxOpacity-self.dOpacity*self.cunning))
            self.fakeDodge = Rectangle(pos=(x+buffer, y+buffer), size=size)
        self.RT = Button(text='Real', pos=(x, y+2*buffer), size_hint=(0.04, 0.04), background_disabled_normal='', background_color=(1, 1, 1, 0.5), color=(0, 0, 0, 1), disabled=True)
        self.FT = Button(text='Fake', pos=(x+buffer, y+2*buffer), size_hint=(0.04, 0.04), background_disabled_normal='', background_color=(1, 1, 1, 0.5), color=(0, 0, 0, 1), disabled=True)
        self.fpage.add_widget(self.RT)
        self.fpage.add_widget(self.FT)
    def npcAttacks(self):
        trials = max([1, self.fpage.foestats[self.fpage.P.attributes['Technique']] * (2 ** (npcDifficulty-3))])
        Atk = np.max(rbtwn(0, self.fpage.foestats[self.fpage.P.attributes['Attack']], trials)) # Attack chosen technique amt of times, then max is chosen
        if Atk > self.fpage.foestats[self.fpage.P.attributes['Stability']]:
            stability_impacted = max([1, npcDifficulty-2])
            if rbtwn(0,stability_impacted)==0:
                # Lack of stability effected foe's attack
                Atk = self.fpage.foestats[self.fpage.P.attributes['Stability']]
        if self.fpage.logDef:
            Atk = max([2, Atk])
        if Atk > 0:
            pagility = self.fpage.pstats[self.fpage.P.attributes['Agility']]
            hitBoxScale = (1 - 0.055*self.fpage.pstats[self.fpage.P.attributes['Stability']])
            hitBox_x, hitBox_y = 0.3*hitBoxScale*self.fpage.size[0], 0.7*hitBoxScale*self.fpage.size[1]
            xshift, yshift = (0.02 + (0.3 - 0.3*hitBoxScale)/2), (0.02 + (0.7 - 0.7*hitBoxScale)/2)
            cunning = max([0, self.fpage.foestats[self.fpage.P.attributes['Cunning']] - rbtwn(0, pagility//2)])
            normalizedCoords = np.random.rand(Atk+cunning, 2) # Random box coordinates are produced length of fakes and reals
            fakes = np.zeros((len(normalizedCoords),),dtype=bool)
            if len(normalizedCoords) > 1:
                # Randomly assign some as fakes with the last being a hit
                fakes[np.random.choice(np.arange(len(normalizedCoords)-1), cunning, False)] = True
            positions, sizes = [], []
            technique = self.fpage.foestats[self.fpage.P.attributes['Technique']]
            trials = max([max([1, npcDifficulty-1]), (2 ** (npcDifficulty-4))])
            centricity = 0.2 + 0.2*(5-npcDifficulty)
            for coord in normalizedCoords:
                shiftedCoord = [coord[0]**centricity if coord[0]>=0.5 else (1 - (1 - coord[0])**centricity),
                                coord[1]**centricity if coord[1]>=0.5 else (1 - (1 - coord[1])**centricity)]
                # Arbitrarily give technique 3 chances to take effect and pick largest outcome
                tApplied = np.max(rbtwn(0, technique, trials))
                size = (self.boxMaxScale - tApplied*self.techniqueScale, self.boxMaxScale - tApplied*self.techniqueScale)
                rawpos = [shiftedCoord[0]*hitBox_x - size[0]/2, shiftedCoord[1]*hitBox_y - size[1]/2] # This will be shifted by 0.02 once technique size is applied
                positions.append((float(rawpos[0])+xshift*self.fpage.size[0], float(rawpos[1])+yshift*self.fpage.size[1]))
                sizes.append(size)
            self.conductDefense(positions, sizes, fakes)
        else:
            # [INCOMPLETE] Display that NPC missed his attack
            self.clearAll()
            self.fpage.msgBoard.text = 'Foe missed!'
            self.fpage.nextAttack()
    def getNormalizedPosition(self, pos):
        xMin, xMax, yMin, yMax = self.fpage.size[0]*0.02, self.fpage.size[0]*0.32, self.fpage.size[1]*0.02, self.fpage.size[1]*0.72
        return np.array([(pos[0] - xMin)/(xMax - xMin), (pos[1] - yMin)/(yMax - yMin)])
    def conductDefense(self, positions, sizes, fakes):
        self.volleys = []
        self.volley_group = {}
        volleys = []
        self.positions = np.zeros((len(positions), 4))
        self.posids = np.arange(len(positions))
        for i in range(len(positions)):
            volleys.append([positions[i], sizes[i], fakes[i], i, len(self.volleys)])
            if not fakes[i]:
                self.volley_group[len(self.volleys)] = [v[3] for v in volleys]
                self.volleys.append(volleys)
                volleys = []
        self.current_volley = 0
        self.live_rects = {}
        Clock.schedule_interval(self.defendVolley, self.strikeDelay)
    def add2Log(self, i):
        R = self.live_rects[i]['rect']
        tApplied = int(-(R.size[0] - self.boxMaxScale)/self.techniqueScale)
        vg = self.live_rects[i]['volley group']
        npos = self.live_rects[i]['nrmpos']
        d2c = essf.euc(npos, np.array([0.5, 0.5]))
        if vg > 0:
            for gi in self.volley_group[vg-1]:
                if not self.live_rects[gi]['fake']:
                    d2p = essf.euc(npos, self.live_rects[gi]['nrmpos'])
                    break
        else:
            d2p = d2c
        d2g, total = 0, 0
        for gi in self.volley_group[vg]:
            if gi == i:
                continue
            d2g += essf.euc(npos, self.live_rects[gi]['nrmpos'])
            total += 1
        if total > 0: d2g /= total
        clicked = 1 if self.live_rects[i]['state'] in {'blocked', 'dodged'} else 0
        dodged = 1 if self.live_rects[i]['state'] == 'dodged' else 0
        with open('data\\DefenseLog.csv', 'a', newline='') as f:
            fcsv = csv.writer(f)
            fcsv.writerow([self.foeagility, self.stability, self.dodging, vg, len(self.volley_group[vg]), tApplied, d2p, d2c, d2g, 0.96 if self.live_rects[i]['fake'] else (self.mxOpacity-self.dOpacity*self.cunning), clicked, dodged])
    def defendVolley(self, instance=None):
        # Make sure we haven't reached the limit of volleys and that the player has not run.
        if (self.current_volley < len(self.volleys)) and (self.fpage.fighting):
            for volley in self.volleys[self.current_volley]:
                with self.fpage.canvas.after:
                    C = Color(0.1, 0.8, 0.1, (self.mxOpacity-self.dOpacity*self.cunning) if volley[2] else 0.96) if self.initState == 'dodge' else Color(0.2, 0.2, 0.6, (self.mxOpacity-self.dOpacity*self.cunning) if volley[2] else 0.96)
                    R = Rectangle(pos=volley[0], size=volley[1])
                self.live_rects[volley[3]] = {'rect':R, 'color':C, 'fake':volley[2], 'state':'dodge' if self.initState == 'dodge' else 'block', 'removed':False, 'volley group':volley[4], 'nrmpos':self.getNormalizedPosition(R.pos)}
                self.positions[volley[3]] = [volley[0][0], volley[0][0]+volley[1][0], volley[0][1], volley[0][1]+volley[1][1]]
                self.transitionBox(volley[3], self.live_rects[volley[3]]['state'])
            self.current_volley += 1
        else:
            self.clearAll()
            self.fpage.nextAttack()
            return False
    def transitionBox(self, i, prevState):
        def transition(_=None):
            if (self.live_rects[i]['state'] == prevState) and (not self.live_rects[i]['removed']):
                if prevState == 'dodge':
                    self.live_rects[i]['state'] = 'block'
                    self.live_rects[i]['color'].rgba = (0.2, 0.2, 0.6, (self.mxOpacity-self.dOpacity*self.cunning) if self.live_rects[i]['fake'] else 0.96)
                    self.transitionBox(i, 'block')
                elif prevState == 'block':
                    self.hitBox(i)
        Clocked(transition, self.dodgeTime if prevState=='dodge' else self.blockTime, 'block/dodge time')
    def removeBox(self, i, _=None):
        def rmvBox(_=None):
            self.live_rects[i]['removed'] = True
            self.fpage.canvas.after.remove(self.live_rects[i]['rect'])
            if self.fpage.logDef:
                self.add2Log(i)
        if not self.live_rects[i]['removed']:
            Clocked(rmvBox, 0.1, 'remove box')
        if not self.live_rects[i]['fake']:
            for group_i in self.volley_group[self.live_rects[i]['volley group']]:
                if i == group_i:
                    continue
                self.revealFake(group_i)
    def revealFake(self, i):
        if self.live_rects[i]['fake']:
            self.live_rects[i]['color'].rgba = (0.9, 0.9, 0.9, 0.6)
            self.removeBox(i)
            return True
        return False
    def dodgeBox(self, i):
        self.live_rects[i]['state'] = 'dodged'
        if not self.revealFake(i):
            self.live_rects[i]['color'].rgba = (0.05, 0.5, 0.05, 1)
            self.removeBox(i)
    def blockBox(self, i):
        self.live_rects[i]['state'] = 'blocked'
        if not self.revealFake(i):
            self.live_rects[i]['color'].rgba = (0.05, 0.05, 0.5, 1)
            self.removeBox(i)
    def hitBox(self, i):
        self.live_rects[i]['state'] = 'hit'
        if not self.revealFake(i):
            self.live_rects[i]['color'].rgba = (0.8, 0.05, 0.05, 1)
            self.removeBox(i)
            self.fpage.pTakesDamage()
    def on_touch_down(self, touch=None):
        if (touch is not None):
            x0 = touch.pos[0] >= self.positions[:,0]
            x1 = touch.pos[0] <= self.positions[:,1]
            y0 = touch.pos[1] >= self.positions[:,2]
            y1 = touch.pos[1] <= self.positions[:,3]
            idsTouched = self.posids[np.all([x0, x1, y0, y1], 0)]
            blockStates = []
            for i in idsTouched:
                if self.live_rects[i]['state'] == 'block':
                    blockStates.append(i)
                elif self.live_rects[i]['state'] == 'dodge':
                    self.dodgeBox(i)
            if (len(blockStates) > 0):
                if (self.blocksRemaining>0):
                    self.blocksRemaining -= 1
                    for i in blockStates:
                        self.blockBox(i)
                else:
                    for i in blockStates:
                        self.hitBox(i)
    def clearAll(self):
        self.fpage.remove_widget(self.RT)
        self.fpage.remove_widget(self.FT)
        self.fpage.canvas.after.remove(self.realBlock)
        self.fpage.canvas.after.remove(self.fakeBlock)
        self.fpage.canvas.after.remove(self.realDodge)
        self.fpage.canvas.after.remove(self.fakeDodge)
        self.fpage.remove_widget(self)


class FightPage(FloatLayout):
    def __init__(self, name, style, lvl, stats, encountered=True, logDef=False, reward=None, consume=None, action_amt=1, foeisNPC=True, background_img=None, consequence=None, foeStealth=0, **kwargs):
        super().__init__(**kwargs)
        logging.debug("Initiatating Fight Page")
        self.logDef = logDef
        self.consume = consume
        self.action_amt = action_amt
        self.reward = reward
        self.consequence = consequence
        self.check_attributes = ['Agility', 'Cunning', 'Hit Points', 'Attack', 'Stability', f'Def-{style}']
        self.foeisNPC = foeisNPC
        self.order_refresh_count = 0
        self.attack_count = 0

        # Player objects
        self.P = lclPlayer()
        self.P.paused = True # prevents player from switching to another page and making an action.
        self.fighting = True

        # Stats
        if self.logDef:
            self.pstats = npc_stats(rbtwn(3, 120))
            self.foestats = npc_stats(rbtwn(3, 120))
            output(f"Activating Logged Def, Player Lv.{np.sum(self.pstats)}, NPC Lv.{np.sum(self.foestats)}")
        else:
            self.pstats, Gsum = deepcopy(self.P.current), np.zeros((len(self.P.attributes),),dtype=int)
            if (len(self.P.group) > 0) and name in {'Duelist', 'Sparring Partner'}:
                output("Your group does not battle with you.", 'yellow')
            else:
                for Name, G in self.P.group.items():
                    output(f"Level {np.sum(G)} {Name} boosts stats. Combined stat is maximum of each attribute.")
                    self.pstats = np.max([self.pstats, G], 0)
                    Gsum += G
            # Apply algorithmic boost for party size - minimum of sum and logged multiplier
            for i in range(len(self.pstats)):
                self.pstats[i] = int(min([self.pstats[i]+Gsum[i], self.pstats[i]*(1 + np.log10(len(self.P.group)+1))]))
            self.foestats = stats
        self.p_startinglvl = np.sum(self.pstats)
        self.f_startinglvl = np.sum(self.foestats)
        # Reduce Stats based on these hard limits
        hard_limits = {"Technique":12, "Stability":14}
        for atr, lvl in hard_limits.items():
            self.pstats[self.P.attributes[atr]] = min([lvl, self.pstats[self.P.attributes[atr]]])
            self.foestats[self.P.attributes[atr]] = min([lvl, self.foestats[self.P.attributes[atr]]])
        logging.info(f"Player Stats: {self.pstats}")
        logging.info(f"Foe Stats: {self.foestats}")
        # Foe objects
        self.foename = name
        self.foelvl = lvl
        self.foestyle = style
        self.foestealth = foeStealth
        self.foecateg = self.foeCategory()
        self.enc = encountered # enc of -1 means the player is triggering the encounter
        # If fighting between people objects
        self.pconfirmed = False
        self.foeconfirmed = False

        # Calculate Disadvantages
        self.p_affected_stats = set()
        self.f_affected_stats = set()
        self.pstats[self.P.attributes["Stability"]] = max([0, self.pstats[self.P.attributes["Stability"]]-self.P.fatigue//3]) # Account for fatigue
        if self.pstats[self.P.attributes["Stability"]] != self.P.current[self.P.attributes["Stability"]]: self.p_affected_stats.add('Stability')
        disadvantages = [['Physical', 'Trooper'], ['Trooper', 'Elemental'], ['Elemental', 'Wizard'], ['Wizard', 'Physical']]
        agil_i = self.P.attributes['Agility']
        for weaker, stronger in disadvantages:
            if (self.P.combatstyle == weaker) and (self.foestyle == stronger):
                self.pstats[agil_i] = max([0, self.pstats[agil_i]-2])
                if self.pstats[agil_i] != self.P.current[agil_i]: self.p_affected_stats.add('Agility')
            elif (self.P.combatstyle == stronger) and (self.foestyle == weaker):
                prior = self.foestats[agil_i]
                self.foestats[agil_i] = max([0, self.foestats[agil_i]-2])
                if self.foestats[agil_i] != prior: self.f_affected_stats.add('Agility')
        # First add background based on tile encounter
        if name == 'Sparring Partner':
            bkgSource = 'images\\resized\\background\\sparringground.png'
        elif background_img is None:
            bkgSource = f'images\\resized\\background\\{self.P.currenttile.tile}.png' if (self.P.currenttile.tile+'.png') in os.listdir('images\\background') else f'images\\resized\\background\\{self.P.currenttile.tile[:-1]}.png'
        else:
            bkgSource = background_img
        self.add_widget(Image(source=bkgSource, pos=(0,0), size_hint=(1,1)))

        # Ideally the ratio will be determined by the ratio of the image with fixed width
        psource = f'images\\resized\\origsize\\{self.P.birthcity}.png'
        x_max, y_max = 0.3, 0.7 # These ratios are also used in resize_images!
        #x_hint, y_hint = self.get_size_hint(x_max, y_max, psource)
        x_hint, y_hint = x_max, y_max

        # Plot Player
        self.pimg = Image(source=psource, pos_hint={'x':0.02, 'y':0.02}, size_hint=(x_hint, y_hint))
        self.add_widget(self.pimg)
        # Plot Foe
        style_num = str(rbtwn(1,3)) if name == 'Duelist' else ''
        fsource = f'images\\resized\\npc\\{name}\\{style}{style_num}.png'
        if not os.path.exists(fsource):
            # Must belong to a city or must be a username
            if foeisNPC:
                for city in cities:
                    if city in self.foename.lower():
                        # They must belong to this city
                        break
                fsource = f'images\\resized\\origsize\\{city}.png'
            else:
                fsource = f'images\\resized\\origsize\\{self.P.parentBoard.Players[self.foename].birthcity}.png'
        x_hint, y_hint = self.get_size_hint(x_max, y_max, fsource)
        self.fimg = Image(source=fsource, pos_hint={'right':0.98, 'y':0.02}, size_hint=(x_hint, y_hint))
        self.add_widget(self.fimg)
        # Generate Table
        self.statTable = self.get_table()
        self.statTable.pos_hint = {'x':0.32, 'y':0.1}
        self.statTable.size_hint = (0.36, 0.7)
        self.add_widget(self.statTable)
        # Message Board
        self.initiateFightOrder()
        p1 = int(np.sum(self.fightorder==self.fightorder[0]))
        p2 = int(np.sum(self.fightorder==self.fightorder[-1]))
        fightorder = f"Fight Order: {p1} time{'s' if p1>1 else ''} {self.fightorder[0]} then {p2} time{'s' if p2>1 else ''} {self.fightorder[-1]}\n"
        msg = f'You are encountered by {self.foename} Lvl. {lvl} using {style} combat style.\n' if encountered else ''
        msg = msg+f"{fightorder}You can't faint but leveling up probability is reduced.\nRun or Fight?" if self.foename == 'Sparring Partner' else msg+f'{fightorder}Run or Fight?'
        self.msgBoard = Button(text=msg, background_color=(1,1,1,0.6), color=(0.3, 0.3, 0.7, 1), background_disabled_normal='', disabled=True, pos_hint={'x':0.32, 'y':0.8}, size_hint=(0.36, 0.1))#, markup=True)
        self.add_widget(self.msgBoard)
        # Run or Fight
        self.runButton = Button(text="", background_color=(1, 1, 0, 1), background_normal='', color=(0,0,0,1), pos_hint={'x':0.32, 'y':0}, size_hint=(0.17, 0.09))
        self.set_runFTG(self.foecateg)
        self.runButton.bind(on_press=self.run)
        category = ['Very Weak', 'Weak', 'Match', 'Strong', 'Very Strong']
        self.fightButton = Button(text=f'Fight ({category[self.foecateg]})', background_color=(0.6, 0, 0, 1), background_normal='', color=(1,1,1,1), pos_hint={'x':0.51, 'y':0}, size_hint=(0.17, 0.09))
        self.fightButton.bind(on_press=self.startBattle)
        self.add_widget(self.runButton)
        self.add_widget(self.fightButton)

        # Determine Stealth Attack
        if (self.foestealth > 0) and encountered and foeisNPC:
            r = rbtwn(1, 12)
            if r <= self.foestealth:
                dmg = self.foestats[self.P.attributes['Agility']] // 3
                if dmg > 0:
                    output(f"Foe inflicts {dmg} to you via a sneak attack!", 'red')
                    self.pTakesDamage(dmg)
        elif (encountered == -1) and foeisNPC and (self.P.skills["Stealth"]>0):
            r = rbtwn(1, 12, None, self.P.skills["Stealth"], "Sneatk Attack ")
            if r <= self.P.skills["Stealth"]:
                dmg = self.pstats[self.P.atrributes['Agility']] // 3
                if dmg > 0:
                    output(f"You inflict {dmg} via a sneak attack!", 'green')
                    self.foeTakesDamage(dmg)
    def foeCategory(self):
        differential = np.sum(self.foestats) - np.sum(self.pstats)
        if differential >= 12:
            return 4
        elif differential >= 6:
            return 3
        elif differential >= -4:
            return 2
        elif differential >= -9:
            return 1
        else:
            return 0
    def set_runFTG(self, ftg):
        self.runFTG = ftg
        self.runButton.text = f'Run (+{ftg} FTG)'
    def run(self, instance=None):
        self.endFight(True)
    def initiateFightOrder(self):
        agility = self.foestats[self.P.attributes["Agility"]], self.pstats[self.P.attributes["Agility"]]
        playeragil = agility[1]+2 if self.enc==-1 else agility[1] # In the case that the player was the one triggering the encounter
        startagil = agility[0]+2 if self.enc else agility[0]
        if startagil > playeragil:
            order = [0]*max([1, int(agility[0]/max([1,agility[1]]))])+[1]*max([1, int(agility[1]/max([1,agility[0]]))])
        elif playeragil > startagil:
            order = [1]*max([1,int(agility[1]/max([1,agility[0]]))])+[0]*max([1, int(agility[0]/max([1,agility[1]]))])
        else:
            # [INCOMPLETE] If this is between two players, then we need a way for them to communicate with each other who goes first -- or a tie breaker: I propose the triggering person brings a random number always just in case
            f = np.random.choice(np.arange(2))
            order = [f]*max([1,int(agility[f]/max([1,agility[1-f]]))]) + [1-f]*max([1,int(agility[1-f]/max([1,agility[f]]))])
        fighters = np.array([self.foename, self.P.username])
        #output("Order of Attack: "+str(list(fighters[order])))
        self.fightorder = fighters[order]
        self.order_idx = -1
    def startBattle(self, instance=None):
        if self.foeisNPC:
            self.confirmBattle()
        else:
            self.pconfirmed = True
            socket_client.send('[FIGHT]', [self.foename, 'confirmed'])
            if self.foeconfirmed:
                self.confirmBattle()
            else:
                self.msgBoard.text = f"Awaiting for {self.foename} to Start Fight"
    def foeconfirm(self, _=None):
        self.foeconfirmed = True
        if self.pconfirmed:
            self.confirmBattle()
        else:
            self.msgBoard.text = f"{self.foename} is Ready to Fight"
    def confirmBattle(self, _=None):
        self.msgBoard.text = ''
        self.remove_widget(self.fightButton)
        self.nextAttack(0)
    def nextAttack(self, delay=1.2, _=None):
        if hasattr(self, 'atkButton'):
            self.remove_widget(self.atkButton)
        def trigger(_=None):
            if self.fighting:
                self.attack_count += 1
                self.order_idx += 1
                if self.order_idx >= len(self.fightorder):
                    # This means a full round was completed, each player loses stability as long as 2 refreshes have been completed (only applied to this fight)
                    self.order_refresh_count += 1
                    if not (self.order_refresh_count % 2):
                        s_i = self.P.attributes['Stability']
                        oldpstat = self.pstats[s_i]
                        self.pstats[s_i] = max([0, self.pstats[s_i] - 1])
                        if oldpstat != self.pstats[s_i]:
                            self.statTable.cells[self.P.username][s_i].background_color = (1, 1, 0.2, 0.6)
                            self.statTable.cells[self.P.username][s_i].text = str(self.pstats[s_i])
                        oldfstat = self.foestats[s_i]
                        self.foestats[s_i] = max([0, self.foestats[s_i] - 1])
                        if oldfstat != self.foestats[s_i]:
                            self.statTable.cells[self.foename][s_i].background_color = (1, 1, 0.2, 0.6)
                            self.statTable.cells[self.foename][s_i].text = str(self.foestats[s_i])
                    # Recycle order index
                    self.order_idx = 0
                self.msgBoard.text = f"{self.fightorder[self.order_idx]} is Attacking Next"
                Clocked(self.triggerNext, delay, 'trigger next attack')
        Clocked(trigger, delay, 'trigger attack trigger')
    def triggerNext(self, _=None):
        if self.fightorder[self.order_idx] == self.P.username:
            self.playerAttacks()
        else:
            self.playerDefends()
    def playerAttacks(self):
        logging.debug("Player is attacking")
        self.set_runFTG(self.foecateg)
        self.msgBoard.text = 'Choose Attack'
        self.curAtk, self.maxAtk = 0, self.pstats[self.P.attributes["Attack"]]
        self.atkButton = Button(text='0', background_color=(2, 1, 1, 1), color=(1,1,1,1), pos_hint={'x':0.02, 'y':0.8}, size_hint=(0.05, 0.05))
        self.atkButton.bind(on_press=self.exitClickLoop)
        self.add_widget(self.atkButton)
        self.unclicked = True
        timeBetween = 0.07 + 0.04 * self.pstats[self.P.attributes["Technique"]]
        Clock.schedule_interval(self.atkSwitchLoop, timeBetween)
    def atkSwitchLoop(self, _=None):
        if self.unclicked:
            if self.curAtk >= self.maxAtk:
                self.curAtk = 0
            else:
                self.curAtk += 1
            self.atkButton.text = str(self.curAtk)
        else:
            self.atkButton.disabled=True
            return False
    def exitClickLoop(self, instance):
        self.unclicked = False
        stability = self.pstats[self.P.attributes['Stability']]
        if self.curAtk > stability:
            if rbtwn(0,self.P.stability_impact):
                self.msgBoard.text = f'Strike with {self.curAtk} {self.P.combatstyle.lower()} attack!'
            else:
                self.curAtk = stability
                strike = 'You missed.' if self.curAtk == 0 else f'Strike with {self.curAtk} {self.P.combatstyle.lower()} attack!'
                self.msgBoard.text = f'Your attack was unstable!\n{strike}'
        else:
            self.msgBoard.text = 'You missed.' if self.curAtk == 0 else f'Strike with {self.curAtk} {self.P.combatstyle.lower()} attack!'
        if self.curAtk > 0:
            logging.debug("Scaling hit box")
            hitBoxScale = (1 - 0.055*self.foestats[self.P.attributes["Stability"]])
            hitBox_x, hitBox_y = 0.3*hitBoxScale, 0.7*hitBoxScale
            self.hitBox = HitBox(self, text='', background_color=(1,0.4,0,0.2), disabled=True, background_disabled_normal='', pos_hint={'center_x':0.83, 'center_y':0.37}, size_hint=(float(hitBox_x), float(hitBox_y)))
            self.add_widget(self.hitBox)
        else:
            self.nextAttack()
    def playerDefends(self):
        self.set_runFTG(self.foecateg + 2)
        if hasattr(self, 'hitBox'):
            self.hitBox.clearAll()
        hitBoxScale = (1 - 0.055*self.pstats[self.P.attributes['Stability']])
        hitBox_x, hitBox_y = 0.3*hitBoxScale, 0.7*hitBoxScale
        self.defBox = DefBox(self, text='', background_color=(0.7,0.7,0.7,0.4), disabled=True, background_disabled_normal='', pos_hint={'center_x':0.17, 'center_y':0.37}, size_hint=(float(hitBox_x), float(hitBox_y)))
        self.add_widget(self.defBox) # May want to move this to within defBox if the attack is greater than 0
        if self.foeisNPC:
            self.defBox.npcAttacks()
        else:
            # [INCOMPLETE] Listen for attack
            pass
    def get_table(self, opacity=0.6):
        header = [self.P.username, "Attributes", self.foename]
        data = []
        for i in range(len(self.P.atrorder)):
            diff = self.pstats[i] - self.foestats[i]
            if isbetween(0, 5, diff):
                bkg = (1 - diff/5, 1, 1 - diff/5, opacity)
            elif diff > 5:
                bkg = (0, 1 - (diff - 5)/14, 0, opacity)
            elif isbetween(-5, 0, diff):
                bkg = (1, 1 + diff/5, 1 + diff/5, opacity)
            else:
                bkg = (1 - (-diff - 5)/14, 0, 0, opacity)
            pbkg = (1, 1, 0.2, opacity) if self.P.atrorder[i] in self.p_affected_stats else (1, 1, 1, opacity) # Bring attention to affected stats
            fbkg = (1, 1, 0.2, opacity) if self.P.atrorder[i] in self.f_affected_stats else (1, 1, 1, opacity)
            data.append([{"text":str(self.pstats[i]), 'disabled':True, 'background_color':pbkg, 'color':(0,0,0,1), 'background_disabled_normal':''},
                         {"text":self.P.atrorder[i], 'disabled':True, 'background_color':bkg, 'color':(0, 0, 0, 1), 'background_disabled_normal':''},
                         {"text":str(self.foestats[i]), 'disabled':True, 'background_color':fbkg, 'color':(0,0,0,1), 'background_disabled_normal':''}])
        return Table(header, data, header_color=(50,50,50), header_as_buttons=True, header_bkg_color=(1, 1, 1, opacity/2))
    def get_size_hint(self, x_max, y_max, imgSource):
        imgSize = pilImage.open(imgSource).size if type(imgSource) is str else imgSource # If imgSource is not string then assume that size has been fixed or already calculated
        y_hint = x_max * (imgSize[1] / imgSize[0]) # Find y ratio
        if y_hint > y_max:
            x_hint = y_max * (imgSize[0] / imgSize[1]) # Instead use the x ratio
            y_hint = y_max
        else:
            x_hint = x_max
        return x_hint, y_hint
    def endFight(self, ran=False):
        self.fighting = False
        self.P.parentBoard.game_page.main_screen.current = "Board" if self.P.parentBoard.game_page.toggleView.text=="Player Track" else "Player Track"
        self.P.parentBoard.game_page.main_screen.remove_widget(self.P.parentBoard.game_page.fightscreen)
        self.P.paused = False
        hp_dmg = self.P.current[self.P.attributes['Hit Points']] - self.pstats[self.P.attributes['Hit Points']]
        if (self.foename == 'Sparring Partner') and (hp_dmg == self.P.current[self.P.attributes['Hit Points']]):
            # You can't faint when practicing, so make the leftover HP = 1
            output("You lost the sparring match!", 'yellow')
            hp_dmg -= 1
        if not ran:
            output("You take an extra fatigue for fighting")
        fainted_or_paralyzed = self.P.takeDamage(hp_dmg, self.runFTG if ran else 1) # Take damage equal to the amount you took
        action_amt = 0 if fainted_or_paralyzed else self.action_amt # We want to make sure that the actions are at least listed even if they are paralyzed or fainted.
        if not fainted_or_paralyzed:
            exitActionLoop(consume=self.consume, amt=action_amt)() # [INCOMPLETE] if fighting player then should be moved randomly
            if (self.foestats[self.P.attributes['Hit Points']] == 0) and (callable(self.reward)):
                # Player must have won -- claim the reward if it was callable
                self.reward()
        elif (ran or (self.pstats[self.P.attributes['Hit Points']] == 0)) and (self.consequence is not None):
            # They must have fainted and lost the battle
            self.consequence()
    def foeTakesDamage(self, amt=1):
        def Reward():
            # Get reward
            if not callable(self.reward):
                for rwd, amt in self.reward.items():
                    amt = self.P.get_bonus(amt)
                    if rwd == 'coins':
                        output(f"Rewarded {int(amt)} coin!", 'green')
                        self.P.coins += int(amt)
                    else:
                        # Assumption: If not coins, then must be an item
                        output(f"Rewarded {int(amt)} {rwd}!", 'green')
                        self.P.addItem(rwd, int(amt))
            # Update title
            self.P.titles['brave']['currentStreak'] += self.foelvl
            if self.P.titles['brave']['currentStreak'] > self.P.titles['brave']['value']:
                self.P.updateTitleValue('brave', self.P.titles['brave']['currentStreak'] - self.P.titles['brave']['value'])
            if self.P.titles['valiant']['value'] < (self.f_startinglvl - self.p_startinglvl):
                self.P.updateTitleValue('valiant', (self.f_startinglvl - self.p_startinglvl) - self.P.titles['valiant']['value'])
            # Training with Sparring partner does not gaurantee you a level increase like with others.
            levelsup = (self.foecateg/2) ** 2
            if self.foename == 'Sparring Partner': levelsup *= 0.65
            levelsup += self.P.combatxp # Get any remaining xp from previous fights
            self.P.combatxp = float(levelsup) - int(levelsup) # Store any remaining xp away
            self.levelsup = int(levelsup)
            if self.levelsup > 0:
                critthink = self.P.activateSkill('Critical Thinking')
                r = rbtwn(1, 12, None, critthink, 'Critical Thinking ')
                if r <= critthink:
                    self.P.useSkill('Critical Thinking')
                    self.prompt_levelup()
                    # prompt_levelup should end the fight once attributes are chosen.
                else:
                    for i in range(self.levelsup):
                        self.check_valid_levelup()
                        if len(self.valid_attributes) == 0:
                            continue
                        random_atr = list(self.valid_attributes.keys())[rbtwn(0, len(self.valid_attributes)-1)]
                        output(f"Leveling up {random_atr}!", 'green')
                        self.P.updateAttribute(random_atr)
                        if self.P.trained_abilities[random_atr]:
                            self.P.trained_abilities[random_atr] = False
                            self.P.updateTitleValue('apprentice', self.P.get_level(random_atr))
                    self.endFight()
            else:
                output(f"You don't level up, level up remainder: {round(self.P.combatxp,2)}", 'yellow')
                self.endFight()
        logging.info(f"Foe is taking damage. Prior HP - {self.foestats[self.P.attributes['Hit Points']]}")
        if (self.foestats[self.P.attributes['Hit Points']] > 0) and (self.fighting):
            self.foestats[self.P.attributes['Hit Points']] = max([0, self.foestats[self.P.attributes['Hit Points']]-amt])
            cell = self.statTable.cells[self.foename][self.P.attributes['Hit Points']]
            cell.text = str(self.foestats[self.P.attributes['Hit Points']])
            cell.background_color = (1, 1, 0.2, 0.6)
            if self.foestats[self.P.attributes['Hit Points']] == 0:
                Reward()
        elif (self.foestats[self.P.attributes['Hit Points']] == 0) and (self.fighting):
            Reward()
    def levelup(self, atr, _=None):
        if self.levelsup > 0:
            output(f"Leveling up {atr}!", 'green')
            self.P.updateAttribute(atr)
            self.levelsup -= 1
            if self.P.trained_abilities[atr]:
                self.P.trained_abilities[atr] = False
                self.P.updateTitleValue('apprentice', self.P.get_level(atr))
            if self.levelsup > 0:
                self.prompt_levelup()
            else:
                self.endFight()
        else:
            self.endFight()
    def check_valid_levelup(self):
        self.valid_attributes = {}
        for atr in self.check_attributes:
            if self.P.combat[self.P.attributes[atr]] < 8:
                self.valid_attributes[atr] = partial(self.levelup, atr)
            elif self.P.trained_abilities[atr]:
                self.valid_attributes[atr] = partial(self.levelup, atr)
    def prompt_levelup(self):
        self.check_valid_levelup()
        if (len(self.valid_attributes) > 0) and (self.levelsup > 0):
            output(f"Choose your level up attributes! ({self.levelsup} Remaining)", 'blue')
            actionGrid(self.valid_attributes, False)
        else:
            self.endFight()
    def pTakesDamage(self, amt=1):
        logging.info(f"Player is taking damage. Prior HP - {self.pstats[self.P.attributes['Hit Points']]}")
        if (self.pstats[self.P.attributes['Hit Points']] > 0) and (self.fighting):
            self.pstats[self.P.attributes['Hit Points']] = max([0, self.pstats[self.P.attributes['Hit Points']]-amt])
            cell = self.statTable.cells[self.P.username][self.P.attributes['Hit Points']]
            cell.text = str(self.pstats[self.P.attributes['Hit Points']])
            cell.background_color = (1, 1, 0.2, 0.6)
            if self.pstats[self.P.attributes['Hit Points']] == 0:
                if self.foename == 'Duelist':
                    self.P.dueling_hiatus = 3 # Losing makes it harder to go back into arena
                self.endFight()

def npc_stats(lvl, fixed=None, maxAtr=None):
    P = lclPlayer()
    lbls = P.atrorder
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
            if atr != 'Stealth':
                # Stealth is handled in the encounter function
                conditions.append(f'a[{P.attributes[atr]}]=={fixedlvl}')
    while True:
        a = essf.randofsum(lvl, n)
        breakout = True
        for i in range(len(conditions)):
            if (conditions[i] is not None) and eval(conditions[i]):
                breakout = False
                break
        if breakout: break
    return a

def encounter(name, lvlrange, styles, reward, fixed=None, party_size=1, consume=None, action_amt=1, empty_tile=False, lvlBalance=6, enc=1, consequence=None, background_img=None):
    P = lclPlayer()
    lvls = []
    for i in range(party_size):
        possibleLvls = rbtwn(lvlrange[0],lvlrange[1],max([1, round((lvlrange[1]-lvlrange[0])/lvlBalance)]))
        lvls.append(possibleLvls[np.argsort(np.abs(possibleLvls-np.sum(P.current)))[0]])
    stat, Gsum, Gmax = np.zeros((len(P.attributes),),dtype=int), np.zeros((len(P.attributes),),dtype=int), np.zeros((len(P.attributes),),dtype=int)
    for i in range(party_size):
        stati = npc_stats(lvls[i])
        Gsum += stati
        Gmax = np.max([Gmax, stati], 0)
    # The Group Level is based on the minumum of the sum and max scores times a logged multiplier
    for i in range(len(stat)):
        stat[i] = int(min([Gsum[i], Gmax[i] * (1 + np.log10(party_size))]))
    lvl = int(np.sum(stat))
    # For rewards go through the dictionary and if value is a list then choose a random result biased toward the level chosen of opponent
    if not callable(reward):
        new_reward = {}
        for rwd, val in reward.items():
            if type(val) is list:
                if party_size == 1:
                    lvlpct = (lvl - lvlrange[0])/(lvlrange[1] - lvlrange[0])
                    val = round(lvlpct * (val[1] - val[0]))
                else:
                    val = rbtwn(val[0], val[1]) # Otherwise just randomize it
            new_reward[rwd] = val
    else:
        new_reward = reward
    cbstyle = styles[rbtwn(0,len(styles)-1)]
    if (fixed is not None) and ("Stealth" in fixed):
        if type(fixed['Stealth']) is list:
            stealth = rbtwn(fixed['Stealth'][0], fixed['Stealth'][1])
        else:
            stealth = fixed['Stealth']
    else:
        stealth = 0
    if party_size==1:
        output(f'Encounter Lv{lvl} {name} using {cbstyle}','yellow')
    else:
        output(f'Encounter {party_size} {name}s with combined level of {lvl} using {cbstyle}', 'yellow')
    screen = Screen(name="Battle")
    screen.add_widget(FightPage(name, cbstyle, lvl, stat, enc, reward=new_reward, consume=consume, action_amt=action_amt, foeisNPC=True, consequence=consequence, background_img=background_img, foeStealth=stealth))
    P.parentBoard.game_page.main_screen.add_widget(screen)
    P.parentBoard.game_page.main_screen.current = "Battle"
    P.parentBoard.game_page.fightscreen = screen

def resize_img(source, scale=3, saveto=None, overwrite=False):
    saveto = os.path.dirname(os.path.dirname(source))+'\\resized\\'+'\\'.join(source.split('\\')[-2:]) if saveto is None else saveto
    if overwrite or (not os.path.exists(saveto)):
        im = pilImage.open(source)
        im_size = im.size
        if type(scale) is tuple:
            float_size = scale
        else:
            float_size = im_size[0]*scale, im_size[1]*scale
        int_size_x, int_size_y = round(float_size[0]), round(float_size[1])
        new_size = (int_size_x, int_size_y)
        im_resized = im.resize(new_size, pilImage.ANTIALIAS)
        im_resized.save(saveto, "PNG")

def resize_images(new_width, new_height):
    gsize = new_width, new_height
    def scale_save(full_imgSource, saveTo):
        im_size = pilImage.open(full_imgSource).size
        mx_x, mx_y = gsize[0]*0.3, gsize[1]*0.7 # These ratios are also used in FightPage!
        scale = min([mx_x / im_size[0], mx_y / im_size[1]]) # For consistent scaling, choose the minumum
        resize_img(full_imgSource, scale, saveTo)
    for imgSource in os.listdir('images\\background'):
        resize_img('images\\background\\'+imgSource, gsize)
    for npc in os.listdir('images\\npc'):
        for imgSource in os.listdir('images\\npc\\'+npc):
            scale_save('images\\npc\\'+npc+'\\'+imgSource, 'images\\resized\\npc\\'+npc+'\\'+imgSource)
    for imgSource in os.listdir('images\\origsize'):
        scale_save('images\\origsize\\'+imgSource, None)

def isbetween(mn, mx, numb):
    return (numb >= mn) * (numb <= mx)

#%% Player to Player Interactions: Fighting

def player_fight(username, stats, encounter):
    P = lclPlayer()
    def consequence():
        if P.item_count > 0:
            L, p = list(P.items), []
            for item in L:
                categ, price = getItemInfo(item)
                p.append(price)
            item = np.random.choice(L, p=np.array(p)/np.sum(p))
            output(f"You lost {item}!", 'red')
            P.addItem(item, -1)
            reward = [item, 1]
        elif P.coins > 0:
            # 30% Coins Taken
            amt = max([1, np.round(0.3*P.coins)])
            reward = ['coins', amt]
            output("You lost {amt} coin{'s' if amt>1 else ''}!",'red')
            P.coins -= amt
        else:
            reward = []
        socket_client.send('[FIGHT]', {username: reward})
    def reward():
        P.player_fight_hiatus[username] = 3
    screen = Screen(name="Battle")
    screen.add_widget(FightPage(username, P.parentBoard.Players[username].combatstyle, int(np.sum(stats)), stats, encounter, reward, consequence=consequence, foeisNPC=False))
    P.parentBoard.game_page.main_screen.add_widget(screen)
    P.parentBoard.game_page.main_screen.current = "Battle"
    P.parentBoard.game_page.fightscreen = screen

def player_attempt_fight(username, excavating, stats, _=None):
    P = lclPlayer()
    def engage_fight(encounter='engaged', _=None):
        socket_client.send('[FIGHT]', {username: encounter, 'stats':P.current})
        enc = -1 if encounter=='encountered' else 0
        player_fight(username, stats, enc)
    def evade_fight(_=None):
        stealth = P.activateSkill("Stealth")
        r = rbtwn(0, int(excavating*(16/12)), None, stealth, "Evasion ")
        if r <= stealth:
            P.useSkill("Stealth")
            output("You successfully evaded the battle!", 'green')
            socket_client.send('[FIGHT]', {username: 'evaded'})
            exitActionLoop(amt=0)()
        else:
            output(f"You are unable to evade! You are encountered by {username}!", 'yellow')
            engage_fight('encountered')
    if P.paused or P.parentBoard.game_page.occupied:
        output(f"{username} attempted to fight you but was rejected due to your status.", 'yellow')
        socket_client.send('[FIGHT]', {username: 'reject'})
        exitActionLoop(amt=0)()
    else:
        output(f"{username} has declared to fight you. What would you like to do?", 'blue')
        actionGrid({"Evade":evade_fight, "Engage":engage_fight}, False)

def player_declare_fight(username, _=None):
    P = lclPlayer()
    if P.paused:
        return
    elif (username in P.player_fight_hiatus) and (P.player_fight_hiatus[username] > 0):
        output(f"You can't fight them for another {P.player_fight_hiatus[username]} rounds", 'yellow')
        return
    output(f"Sending fight declaration to {username}", 'blue')
    socket_client.send('[FIGHT]', {username: 'declare', 'excavating': P.skills["Excavating"], 'stats': P.current})

#%% Action Loops
def output(message, color=None):
    game_app.game_page.update_output(message, color)
    logging.info(message)
def actionGrid(funcDict, save_rest, occupied=True, add_back=False):
    game_app.game_page.make_actionGrid(funcDict, save_rest, occupied, add_back=add_back)
def check_paralysis():
    P = lclPlayer()
    paralyzed = False
    if (P.fatigue > P.max_fatigue) or (P.paralyzed_rounds in {1,2}):
        paralyzed = True
        if P.PlayerTrack.Quest.quests[3, 6].status == 'started':
            P.PlayerTrack.Quest.update_quest_status((3, 6), 'failed')
        P.paralyzed_rounds += 1
        A = np.arange(P.actions)
        for a in A:
            P.recover(None)
        if P.paralyzed_rounds > 2:
            paralyzed = False
            P.paralyzed_rounds = 0
            # If the player is in a city they are not allowed in then move away onto a random road after healed.
            if (P.currenttile.tile in P.entry_allowed) and (not P.entry_allowed[P.currenttile.tile]):
                output("You are kicked out of the city after recovery.",'yellow')
                roadtiles = []
                for coord in P.currenttile.neighbors:
                    if P.currenttile.parentBoard.gridtiles[coord].tile == 'road':
                        roadtiles.append(coord)
                P.moveto(roadtiles[np.random.choice(np.arange(len(roadtiles)))], trigger_consequence=False)
    return paralyzed
def exitActionLoop(consume=None, amt=1, empty_tile=False):
    P = lclPlayer()
    def exit_loop(_=None):
        if P is not None:
            # Player just finished a consequence directly at the start of the round
            if P.started_round:
                if (consume is None) or ('tiered' not in consume):
                    P.go2action()
                else:
                    # Otherwise the player must be descended into a cave or ontop of a mountain
                    tier = int(consume[-1])
                    P.tiered = tier if tier > 1 else False
                    P.go2action(tier)
                P.started_round = False
                if game_launched[0]: logging.info(f"Taking {amt} in starting action!")
            else:
                if consume == 'road':
                    P.road_moves -= amt
                    if P.road_moves == 0:
                        P.takeAction(amt)
                    else:
                        P.update_mainStatPage()
                elif consume == 'minor':
                    P.minor_actions -= amt
                    if P.minor_actions == 0:
                        # If the player makes his max transactions, then it is the same as if he bartered. Otherwise, bartering is handled in the takeAction method.
                        P.takeAction(amt)
                    else:
                        P.update_mainStatPage()
                elif consume == 'purchase':
                    if P.first_purchase_after_barter:
                        # In other words, regardless of if the amt=1 or amt=0, they must take one minor action this round.
                        P.first_purchase_after_barter = False
                        exitActionLoop('minor', amt=1, empty_tile=empty_tile)()
                    else:
                        exitActionLoop('minor', amt=amt, empty_tile=empty_tile)()
                elif amt>0:
                    if game_launched[0]: logging.info(f"Taking {amt} of fatigue!")
                    P.takeAction(amt)
                else:
                    P.update_mainStatPage()
                if empty_tile: P.currenttile.empty_tile()
                if (consume is None) or ('tiered' not in consume):
                    P.go2action()
                else:
                    # Otherwise the player must be descended into a cave or ontop of a mountain
                    tier = int(consume[-1])
                    P.tiered = tier if tier > 1 else False
                    P.go2action(tier)
    return exit_loop

#%% Trading
def getItem(item, amt=1, consume=None, empty_tile=False, action_amt=1):
    P = lclPlayer()
    def getitem(_=None):
        if P.paused:
            return
        P.addItem(item, amt)
        if (P.PlayerTrack.Quest.quests[3, 2].status == 'started') and (P.currenttile.tile == 'mountain'):
            # Assumption is that this function is only called in a mountain if they excavated for ore
            categ, price = getItemInfo(item)
            if categ == 'Smithing':
                P.PlayerTrack.Quest.quests[3, 2].total_ore_cost += sellPrice[price]
                output(f"So far, found ores costing a total of {P.PlayerTrack.Quest.quests[3, 2].total_ore_cost}", 'blue')
                if (P.PlayerTrack.Quest.quests[3, 2].total_ore_cost >= 5) and (P.PlayerTrack.Quest.quests[3, 2].reached_top):
                    P.PlayerTrack.Quest.quests[3, 2].completed_mountain = True
                    output("Now go to the plains and find at least 5 pieces of meat!", 'blue')
        exitActionLoop(consume, action_amt, empty_tile)()
    return getitem

def sellItem(item, to_trader, barter=None, _=None):
    P = lclPlayer()
    if P.paused:
        return
    if item in P.unsellable:
        output("You can't sell this item in the same round!", 'yellow')
        return
    categ, price = getItemInfo(item)
    if categ == 'Cloth':
        if P.currenttile.tile in item:
            # The city for which the cloth belongs can't be sold
            output("The market won't buy that item here!", 'yellow')
            return
        sellprice = price
    else:
        sellprice = sellPrice[price]
    if (barter is None) and (not P.activated_bartering):
        output("Would you like to barter?",'blue')
        actionGrid({'Yes':partial(sellItem, item, to_trader, True), 'No':partial(sellItem, item, to_trader, False), 'Cancel':exitActionLoop(amt=0)}, False, False)
    elif barter == True:
        P.activated_bartering = True
        bartering = P.activateSkill("Bartering")
        r = rbtwn(1, 12, None, bartering, 'Bartering ')
        if r <= bartering:
            P.useSkill("Bartering")
            output("You successfully Barter", 'green')
            if (bartering > 8) and to_trader:
                P.bartering_mode = 2
                sellprice += 2
                P.coins += sellprice
                P.updateTitleValue('merchant', sellprice)
                if to_trader:
                    if P.currenttile.trader_id not in P.titles['trader']['unique_traders']:
                        P.titles['trader']['unique_traders'].add(P.currenttile.trader_id)
                        P.updateTitleValue('trader', 1)
            else:
                P.bartering_mode = 1
                sellprice += 1
                P.coins += sellprice
                P.updateTitleValue('merchant', sellprice)
                if to_trader:
                    if P.currenttile.trader_id not in P.titles['trader']['unique_traders']:
                        P.titles['trader']['unique_traders'].add(P.currenttile.trader_id)
                        P.updateTitleValue('trader', 1)
            # Its basically inverted getItem
            output(f"Sold {item} for {sellprice}.")
            getItem(item, -1, 'minor', False, 1)()
        else:
            output("You failed to barter, sell anyway?", 'yellow')
            actionGrid({'Yes':partial(sellItem, item, to_trader, False), 'No':exitActionLoop('minor', 0, False)}, False, False)
    else:
        sellprice += P.bartering_mode
        output(f"Sold {item} for {sellprice}.")
        P.coins += sellprice
        P.updateTitleValue('merchant', sellprice)
        if to_trader:
            if P.currenttile.trader_id not in P.titles['trader']['unique_traders']:
                P.titles['trader']['unique_traders'].add(P.currenttile.trader_id)
                P.updateTitleValue('trader', 1)
        getItem(item, -1, 'minor', False)()

def buyItem(item, cost, from_trader, amt=1, consume='minor'):
    P = lclPlayer()
    def buyitem(_=None):
        if P.paused:
            return
        if P.coins < (cost*amt):
            output("Insufficient funds!",'yellow')
        elif P.item_count >= P.max_capacity:
            output("Cannot add anymore items!", 'yellow')
        else:
            P.unsellable.add(item)
            P.coins -= int(cost*amt)
            if amt == 1:
                output(f"Bought {item} for {cost} coins")
            else:
                output("Bought {amt} {item} for {cost*amt}")
            if from_trader:
                if P.currenttile.trader_id not in P.titles['trader']['unique_traders']:
                    P.titles['trader']['unique_traders'].add(P.currenttile.trader_id)
                    P.updateTitleValue('trader', 1)
            if from_trader and (item in P.currenttile.trader_wares):
                P.currenttile.buy_from_trader(item)
            getItem(item, amt, consume, False, 1)()
    return buyitem

def Trading(trader, _=None):
    P = lclPlayer()
    if P.paused:
        return
    items = P.currenttile.trader_wares if trader else P.currenttile.city_wares
    if (P.currenttile.tile == 'benfriege') and (P.PlayerTrack.Quest.quests[2, 1].status == 'started'):
        items = items.add('cooling cubes')
    def activate_bartering(_):
        if P.paused:
            return
        P.activated_bartering = True
        bartering = P.activateSkill('Bartering')
        r = rbtwn(1, 12, None, bartering, 'Bartering ')
        if r <= bartering:
            P.useSkill('Bartering')
            output("You successfully Barter", 'green')
            if (bartering > 8) and trader:
                P.bartering_mode = 2
                Trading(trader) # Reduce prices by max of 2
            else:
                P.bartering_mode = 1
                Trading(trader)
        else:
            output("You fail to Barter", 'yellow')
            P.bartering_mode = 0
            Trading(trader) # Not given the opportunity to barter again unless performs a different action first
    actions = {}
    for item in items:
        categ, price = getItemInfo(item)
        cost = price if trader else mrktPrice[price]
        # Reduce cost if have discount in the city
        if (P.currenttile.tile == P.birthcity) and (cost >= P.city_discount_threshold):
            cost = max([1, cost - P.city_discount])
        # Reduce cost depending on trader and bartering mode, but min cost must be 1.
        reduce = P.bartering_mode if trader else min([1, P.bartering_mode]) + (capital_info[P.currenttile.tile]['discount'])
        cost = max([cost - reduce, 1])
        clr = 'ffff75' if P.bartering_mode == 0 else '00ff75'
        actions[f'{item}:[color={clr}]{cost}[/color]'] = buyItem(item, cost, trader, consume='minor')
        if trader and (P.PlayerTrack.Quest.quests[4, 4].status == 'started'):
            if not hasattr(P.PlayerTrack.Quest.quests[4, 4], 'count'):
                P.PlayerTrack.Quest.quests[4, 4].count = 0
            actions["Pursuade"] = PursuadeTrader
        # Give the player the option to go back to regular action menu of the tile
        actions['Back'] = exitActionLoop(None, 0)
    # If the player has not activated bartering already, give them a chance to do so
    if (P.bartering_mode == 0) and (P.activated_bartering==False): actions['Barter'] = activate_bartering
    actionGrid(actions, False)

#%% Item Consumption
heatableItems = {'raw meat':0, 'raw fish':0, 'hide':5, 'clay':6, 'sand':9, 'bark':12}
noncraftableItems = {'hide', 'clay', 'sand'}

def heatitem(item, _=None):
    P = lclPlayer()
    if P.paused:
        return
    if item not in heatableItems:
        output(f"{item} cannot be heated!", 'yellow')
        return
    elif P.skills["Heating"] < heatableItems[item]:
        output(f"You need at least Lv{heatableItems[item]} Heating to attempt the heating!", 'yellow')
        return
    if item not in P.items:
        output(f"You don't own {item} to heat it!", 'yellow')
        return
    heating = P.activateSkill("Heating")
    amtReceived = int(rbtwn(0, heating)/4 + 1)
    recvd, secondaryItem = None, None
    if item == 'raw meat':
        if heating == 0:
            if rbtwn(0,1):
                recvd = 'cooked meat'
        else:
            if heating >= 3:
                # At level 3, hide can be extracted
                secondaryItem = 'hide'
            if heating < 7:
                recvd = 'cooked meat'
            elif isbetween(7, 9, heating):
                recvd = 'well cooked meat' if rbtwn(0,1) else 'cooked meat'
            else:
                recvd = 'well cooked meat'
    elif item == 'raw fish':
        if heating < 2:
            if rbtwn(0,1):
                recvd = 'cooked fish'
        else:
            if heating >= 4:
                # At level 4, scales can be extracted
                secondaryItem = 'scales'
            if heating < 8:
                recvd = 'cooked fish'
            elif isbetween(8, 10, heating):
                recvd = 'well cooked fish' if rbtwn(0,1) else 'coked fish'
            else:
                recvd = 'well cooked fish'
    elif (item == 'hide') and (heating >= 5):
        recvd = 'leather'
    elif (item == 'clay') and (heating >= 6):
        recvd = 'ceramic'
    elif (item == 'sand') and (heating >= 9):
        recvd = 'glass'
    elif (item == 'bark') and (heating >= 12):
        recvd = 'bark'
    if secondaryItem is not None:
        output(f"You extract {secondaryItem}!", 'green')
        P.addItem(secondaryItem, 1)
    if recvd is None:
        output("You destroy the item!", 'red')
    else:
        output(f"You produce {amtReceived} of {recvd}",'green')
        P.addItem(recvd, amtReceived)
    P.addItem(item, -1)
    exitActionLoop('minor')()

def readbook(book, _=None):
    P = lclPlayer()
    if P.paused:
        return
    if book not in P.items:
        output("You don't own this book to read it!", 'yellow')
        exitActionLoop(amt=0)()
        return
    skill = book2skill(book)
    if P.skills[skill] >= 12:
        output(f"You have already maxed out {skill}.", 'yellow')
        exitActionLoop(amt=0)()
        return
    elif P.skills[skill] >= 8:
        output("You cannot level beyond lvl 8 with books. Go seek a trainer.", 'yellow')
        exitActionLoop(amt=0)()
        return
    critthink = P.activateSkill("Critical Thinking")
    r = rbtwn(0, 8, None, critthink, 'Critical Thinking ')
    if r <= critthink:
        P.useSkill("Critical Thinking")
        output(f"You successfully read the {book}!", 'green')
        P.addItem(book, -1)
        P.addXP(skill, P.standard_read_xp)
    else:
        output("You failed to understand. You can try again.", 'red')
    P.fellowships['benfriege'].update_field('read_action_counter', 3)
    exitActionLoop('minor')()

#%% Hallmark Functions
    
def confirm_learn_book(level, skill, _=None):
    P = lclPlayer()
    if P.paused:
        return
    cost = var.library_level_map[level]['cost']
    xp = var.library_level_map[level]['xp']
    book = f'{level} {skill}'.lower()
    P.addItem(book, -1, skip_update=True)
    P.grand_library['borrowed'][book] -= 1
    if P.grand_library['borrowed'][book] <= 0:
        P.grand_library['borrowed'].pop(book)
    P.addItem('learned library book', 1, skip_update=True)
    if 'learned library book' in P.grand_library['borrowed']:
        P.grand_library['borrowed']['learned library book'] += 1
    else:
        P.grand_library['borrowed']['learned library book'] = 1
    P.coins -= cost
    P.addXP(skill, xp)
    exitActionLoop('minor')()

def learn_book(level, skill, _=None):
    P = lclPlayer()
    if P.paused:
        return
    cost = var.library_level_map[level]['cost']
    xp = var.library_level_map[level]['xp']
    if P.coins < cost:
        output(f"You do not have enough coin! It costs {cost} to learn.", 'yellow')
        return
    elif P.skills[skill] < var.library_level_map[level]['skill'][0]:
        output(f'Your {skill} level is too low to learn from this {level} book!', 'yellow')
        return
    elif P.skills[skill] > var.library_level_map[level]['skill'][1]:
        output(f'Your {skill} level is too high to learn from this {level} book!', 'yellow')
        return
    output(f"Do you wish to gain {xp} in {skill} at the cost of {cost} coins?", 'blue')
    actionGrid({'Yes': partial(confirm_learn_book, level, skill)}, False, add_back=True)
    

#%% Training
def Train(abilities, master, confirmed, _=None):
    P = lclPlayer()
    if P.paused:
        return
    usingLesson = True if (abilities == 'stealth') and (master == 'city') and hasattr(P.PlayerTrack.Quest.quests[5, 7], 'used_lesson') and (not P.PlayerTrack.Quest.quests[5, 7].used_lesson) and (P.PlayerTrack.Quest.quests[2, 8].coord == P.currentcoord) else False
    requirement = 'go to combat' if abilities in P.attributes else 'use the skill successfully'
    if not confirmed:
        if master == 'monk':
            cost = 0
        elif usingLesson:
            cost = 0
        elif type(abilities) is str:
            if P.get_level(abilities) >= 12:
                output(f"You have already maxed out {abilities}", 'yellow')
                exitActionLoop(amt=0 if master in {'adept', 'city'} else 1)()
                return
            elif master == 'adept':
                cost = 0 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else P.get_level(abilities)+5
            elif master == 'city':
                cost = (P.get_level(abilities)+11)//2 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else P.get_level(abilities)+11
            else:
                cost = 8 if P.get_level(abilities) < 8 else 12
        else:
            costs, leftoverabilities = [], []
            for ability in abilities:
                if P.get_level(ability) >= 12:
                    continue
                elif master == 'adept':
                    cost = 0 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else P.get_level(abilities)+5
                elif master == 'city':
                    cost = (P.get_level(abilities)+11)//2 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else P.get_level(abilities)+11
                else:
                    cost = 8 if P.get_level(ability) < 8 else 12
                costs.append(f'{cost} for {ability}')
                leftoverabilities.append(ability)
            if len(costs) == 0:
                output("You have already maxed out these abilities", 'yellow')
                exitActionLoop(amt=0 if master in {'adept', 'city'} else 1)()
                return
            cost = '('+', '.join(costs)+')'
            abilities = leftoverabilities
        output(f"Would you like to train at cost of {cost} coins?", 'blue')
        game_app.game_page.make_actionGrid({'Yes':(lambda _:Train(abilities, master, True)), 'No':exitActionLoop(amt=0 if master in {'adept','city'} else 1)}, False, False if master in {'adept', 'city'} else True)
    elif type(abilities) is not str:
        output("Which skill would you like to train?", 'blue')
        game_app.game_page.make_actionGrid({ability:(lambda _:Train(ability, master, confirmed)) for ability in abilities})
    else:
        lvl = P.get_level(abilities)
        if lvl >= 12:
            output(f"You have already maxed out {abilities}", 'yellow')
            exitActionLoop(amt=0 if master in {'adept', 'city'} else 1)()
        elif master == 'adept':
            cost = 0 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else lvl+5
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
                newlevel = P.levelup(abilities,1,8)
                P.updateTitleValue('apprentice', newlevel)
                exitActionLoop(None,1)()
            else:
                critthink = P.activateSkill('Critical Thinking')
                if rbtwn(1,12) <= critthink:
                    P.useSkill('Critical Thinking')
                    output(f"You successfully leveled up in {abilities}",'green')
                    newlevel = P.levelup(abilities,1,8)
                    P.updateTitleValue('apprentice', newlevel)
                    exitActionLoop(None,1)()
                else:
                    output("You were unsuccessful.",'red')
                    P.takeAction()
                    Train(abilities, master, False)
        else:
            if master == 'city':
                if usingLesson:
                    cost = 0
                else:
                    cost = (lvl+11)//2 if (P.training_discount and (P.currenttile.tile==P.birthcity)) else lvl+11
            elif master == 'monk':
                cost = 0
            else:
                cost = 10
            if P.coins < cost:
                output("You do not have sufficient funds.",'yellow')
                # The assumption is that if they are training in city they did not just move into the tile.
                exitActionLoop(None,0 if master=='city' else 1)()
            elif rbtwn(2 if master=='city' else 6,10) <= P.fatigue:
                output("You were unable to keep up with training.",'red')
                P.takeAction()
                Train(abilities, master, False)
            elif lvl < 8:
                output(f"You successfully leveled up in {abilities}",'green')
                P.levelup(abilities,1)
                if usingLesson: P.PlayerTrack.Quest.quests[5, 7].used_lesson = True
                exitActionLoop(None,1)()
            elif P.trained_abilities[abilities]:
                output(f"You have already unlocked the potential to level up {abilities}, now you need to {requirement} to level up!", 'blue')
                exitActionLoop(None,0 if master=='city' else 1)()
            elif rbtwn(1,4) == 1:
                output(f"You successfully unlocked the potential to level up {abilities}! Now {requirement} to level up!", 'green')
                # output(f"You successfully leveled up in {abilities}",'green')
                # P.levelup(abilities,1)
                P.trained_abilities[abilities] = True
                if usingLesson: P.PlayerTrack.Quest.quests[5, 7].used_lesson = True
                exitActionLoop(None,1)()
            else:
                critthink = P.activateSkill('Critical Thinking')
                if rbtwn(1,16) <= critthink:
                    P.useSkill('Critical Thinking', 2, 7)
                    output(f"You successfully unlocked the potential to level up {abilities}! Now {requirement} to level up!", 'green')
                    # output(f"You successfully leveled up in {abilities}",'green')
                    # P.levelup(abilities,1)
                    P.trained_abilities[abilities] = True
                    if usingLesson: P.PlayerTrack.Quest.quests[5, 7].used_lesson = True
                    exitActionLoop(None,1)()
                else:
                    output("You were unsuccessful.",'red')
                    P.takeAction()
                    Train(abilities, master, False)

#%% Consequences
def C_road(action=1):
    P = lclPlayer()
    coord, depth, path = P.currenttile.findNearest(cities)
    r = np.random.rand()
    if P.PlayerTrack.Quest.quests[2, 6].status == 'started': depth *= 2 # Probability of finding robber is doubled
    if r <= (depth/6):
        if (P.PlayerTrack.Quest.quests[2, 3].status == 'started'):
            # Stealth is set to 0
            stealth = 0
        else:
            stealth = P.activateSkill("Stealth")
        r = rbtwn(1, 12, None, stealth, 'Stealth ')
        if r <= stealth:
            P.useSkill('Stealth')
            output("Successfully avoided robber!", 'green')
            exitActionLoop('road')()
        else:
            output("Unable to avoid robber!", 'yellow')
            if P.currenttile.trader_rounds > 0: P.currenttile.remove_trader() # If trader exists on tile, the trader runs away.
            encounter('Highway Robber',[3,30],['Physical','Trooper'],{'coins':[0,3]}, consume=None, action_amt=action)
    elif P.currenttile.trader_rounds == 0:
        r = rbtwn(1, 6)
        if r == 1:
            # Trader appears with 1/6 chance
            P.currenttile.trader_appears()
        exitActionLoop('road')()
    else:
        exitActionLoop('road')()

def C_plains():
    P = lclPlayer()
    survival = P.activateSkill("Survival")
    r = rbtwn(0, 8, None, survival, 'Survival ')
    if r <= survival:
        P.useSkill('Survival')
        output('Successfully avoided traps','green')
        exitActionLoop()()
    else:
        output('Trap hits you, +1 fatigue', 'red')
        paralyzed = P.takeDamage(0,1)
        # If the player is paralyzed then they should still get the actions listed.
        action_amt = 0 if paralyzed else 1
        exitActionLoop(amt=action_amt)()


def C_cave(tier=1, skip=False):
    P = lclPlayer()
    def enemy_approach(_=None):
        stealth = P.activateSkill('Stealth')
        r = rbtwn(0, 4*tier, None, stealth, 'Stealth ')
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
        r = rbtwn(0, 2**tier, None, survival, 'Survival ')
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
        r = rbtwn(0, 12, None, stealth, 'Stealth ')
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
    r = rbtwn(0, 2**tier, None, survival, 'Survival ')
    fainted_or_paralyzed = False
    if r <= survival:
        P.useSkill('Survival',1,2*tier-1)
        output('Successfully traversed harsh environment','green')
    else:
        output(f'Buffeted by harsh environment: -{tier} HP, -{tier} fatigue','red')
        fainted_or_paralyzed = P.takeDamage(tier, tier)
    action_amt = 0 if fainted_or_paralyzed else 1
    exitActionLoop(f'tiered {tier}', amt=action_amt)()

def C_oldlibrary(skip=False):
    def hermit_approach(_=None):
        P = lclPlayer()
        persuasion = P.activateSkill('Persuasion')
        if (P.PlayerTrack.Quest.quests[2, 5].status == 'started') and (P.PlayerTrack.Quest.quests[2, 5].coord == P.currentcoord) and (not P.PlayerTrack.Quest.quests[2, 5].has_book):
            r = rbtwn(1, 5, None, persuasion, 'Persuasion ')
            if r <= persuasion:
                P.useSkill('Persuasion')
                output('Successfully persuaded the hermit to show you the book. Go hand the book to the librarian.', 'blue')
                P.PlayerTrack.Quest.quests[2, 5].has_book = True
                P.parentBoard.gridtiles[P.PlayerTrack.Quest.quests[2, 5].coord].color = (1, 1, 1, 1)
            else:
                output("Could not persuade the hermit to show you the book.", 'yellow')
            exitActionLoop()()
        else:
            r = rbtwn(1, 12, None, persuasion, 'Persuasion ')
            if r <= persuasion:
                P.useSkill('Persuasion')
                output('Successfully persuaded the hermit to teach you!','green')
                Train(['Cunning','Critical Thinking'],'hermit',False)
            else:
                output('Unsuccessful in persuasion','red')
                encounter('Hermit',[55,75],['Wizard','Elemental'],{'coins':[5, 9]},{'Hit Points':12})
    if not skip:
        r = rbtwn(1, 4)
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
        r = rbtwn(1, 12, None, survival, 'Survival ')
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
        r = rbtwn(1, 12, None, persuasion, 'Persuasion ')
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
        r = rbtwn(1, 4)
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
        r = rbtwn(1, 16, None, stealth, 'Stealth ')
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
    r = rbtwn(1, 14, None, survival, 'Survival ')
    if r <= survival:
        P.useSkill("Survival",2,7)
        output("Survived difficult terrain",'green')
        exitActionLoop()()
    else:
        output("Took 3 fatigue and 2 HP damage from terrain",'red')
        fainted_or_paralyzed = P.takeDamage(2, 3)
        if not fainted_or_paralyzed:
            if (P.PlayerTrack.Quest.quests[5, 3].status == 'started') and (not hasattr(P.PlayerTrack.Quest.quests[5, 3], 'killed')):
                def reward(_=None):
                    P.addItem('bark', P.get_bonus(5))
                    P.PlayerTrack.Quest.quests[5, 3].killed = True
                    output("Go claim your reward from the mayor!", 'blue')
                def conseq(_=None):
                    P.PlayerTrack.Quest.update_quest_status((5, 3), 'failed')
                encounter('Great Wild Vine Monster',[135,135],['Elemental'],reward,consequence=conseq)
            else:
                stealth = P.activateSkill("Stealth")
                r = rbtwn(1, 14, None, stealth, 'Stealth ')
                if r <= stealth:
                    P.useSkill("Stealth",2,7)
                    output("Avoided dangerous wilderness monster",'green')
                    exitActionLoop()()
                else:
                    encounter('Wild Vine Monster',[95,120],['Elemental'],{'bark':4})
        else:
            exitActionLoop(amt=0)()

consequences = {'road':C_road, 'plains':C_plains, 'cave':C_cave, 'outpost':C_outpost, 'mountain':C_mountain, 'oldlibrary':C_oldlibrary, 'ruins':C_ruins, 'battle1':C_battlezone, 'battle2':C_battlezone, 'wilderness':C_wilderness}
consequence_message = {'road': 'Highway Robber encounter chance: (distance to nearest city)/6. Highway Robber Lvl: 3-30. Reward: 0-3 coins. Highway Robber Avoidance: sneak/12.\n\nTrader Chance: 1/6 if no Highway Robber encounter.',
                       'plains': 'Fatigue Damage: 1. Damage Avoidance: (survival+1)/9.',
                       'cave': 'HP Damage: 1 in tier1, 3 in tier2, 5 in tier3. Avoidance: (survival+1)/3 in tier1, (survival+1)/5 in tier2, (survival+1)/9 in tier3.\n\nMonster Lvls: 3-20 tier1, 15-40 tier2, 35-60 tier3. Reward: raw meat and hide (varying). Monster Avoidance: (sneak+1)/(4*tier).',
                       'outpost': 'Bandit encounter chance: 1/3. Bandit Lvl: 15-40. Reward: 2-5 coins. Bandit Avoidance: stealth/12.',
                       'mountain': 'HP & Fatigue Damage: tier. Damage Avoidance: (survival+1)/3 in tier1. (surval+1)/5 in tier2, (survival+1)/9 in tier3.',
                       'oldlibrary': 'Hermit encounter chance: 1/4. Hermit Lvl: 55-75. Reward: 5-9 coins. Hermit Avoidance: persuasion/12.',
                       'ruins': 'HP Damage: 2. Damage Avoidance: survival/12.\n\nOld Wizard encounter Lvl 60-85. Reward: 2 old cloth. Old Wizard Avoidance: persuasion/12.',
                       'battle1': 'Ninjas: 1/4 chance 1 Ninja Lvl 65-90, 1/4 chance 1 Ninja Lvl 45-70, 1/4 chance 2 Ninjas Lvls 35-60, 1/4 chance 3 Ninjas Lvls 25-50. Reward: varying coins. Ninja Avoidance: stealth/16.',
                       'battle2': 'Ninjas: 1/4 chance 1 Ninja Lvl 65-90, 1/4 chance 1 Ninja Lvl 45-70, 1/4 chance 2 Ninjas Lvls 35-60, 1/4 chance 3 Ninjas Lvls 25-50. Reward: varying coins. Ninja Avoidance: stealth/16.',
                       'wilderness': 'HP Damage: 2, Fatigue Damage: 3. Damage Avoidance: survival/14.\n\nWild Vine Monster Lvl 95-120. Reward: 4 bark. Wild Vine Monster Avoidance: stealth/14.'}

#%% Actions
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
        if (P.PlayerTrack.Quest.quests[5, 6].status == 'started') and (P.currentcoord == P.PlayerTrack.Quest.quests[1, 8].coord_found) and (not hasattr(P.PlayerTrack.Quest.quests[5, 6], 'killed')):
            excavating = P.activateSkill("Excavating")
            r = rbtwn(1, 30, None, excavating, "Finding Angry Mammoth ")
            if r <= excavating:
                def reward(_=None):
                    output("You killed the Mammoth! Go claim your reward from the mayor.", 'blue')
                    P.PlayerTrack.Quest.quests[5, 6].killed = True
                    P.parentBoard.gridtiles[P.PlayerTrack.Quest.quests[1, 8].coord_found].color = (1, 1, 1, 1)
                encounter("Mammoth", [150, 150], ['Physical'], reward)
                return
            else:
                output("Unable to find the Mammoth", 'yellow')
        gathering = P.activateSkill('Gathering')
        gathered = 1
        while (gathered<6) and (rbtwn(1,12)<=gathering):
            if gathered == 1:
                P.useSkill('Gathering')
            gathered += 1
        output(f"Gathered {gathered} {item}",'green')
        P.addItem(item, gathered)
        if (P.PlayerTrack.Quest.quests[3, 2].status == 'started') and (P.PlayerTrack.Quest.quests[3, 2].completed_mountain) and (P.currenttile.tile == 'plains') and (item == 'raw meat'):
            P.PlayerTrack.Quest.quests[3, 2].meat_collected += gathered
            output(f"So far, gathered a total of {P.PlayerTrack.Quest.quests[3, 2].meat_collected} raw meat", 'blue')
            if P.PlayerTrack.Quest.quests[3, 2].meat_collected >= 5:
                finishFitnessTraining()
        exitActionLoop(None, 1)()
    return gather

def Excavate(result, max_result):
    P = lclPlayer()
    def excavate(_=None):
        if P.paused:
            return
        if (P.PlayerTrack.Quest.quests[5, 2].status == 'started') and (P.currentcoord == P.PlayerTrack.Quest.quests[5, 2].coord):
            output(f"You found the friend! Now go back to {P.birthcity} with him and claim your reward.", 'blue')
            P.PlayerTrack.Quest.quests[5, 2].has_friend = True
            exitActionLoop()()
            return
        elif (P.PlayerTrack.Quest.quests[5, 2].status == 'started'):
            output("Friend is not here", 'yellow')
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

def A_road(inspect=False):
    if inspect:
        return {}, None
    P = lclPlayer()
    if P.currenttile.trader_rounds > 0:
        actions = {'Trader': partial(Trading, True)}
        actionGrid(actions, True)
    else:
        # Otherwise clear the action grid except rest stuff
        actionGrid({}, True)

def A_plains(inspect=False):
    exc = {'Huntsman':[1,1,persuade_trainer(['Agility','Gathering'],'huntsman',exitActionLoop())],
           'Wild Herd':[2,6,Gather('raw meat',0)]}
    if inspect:
        return exc, 9
    else:
        P = lclPlayer()
        def FindBabyMammoth(_=None):
            output("You found a baby mammoth! Its attracted to your fruit and follows you. Don't lose your fruit!", 'green')
            P.PlayerTrack.Quest.quests[1, 8].has_mammoth = True
            P.PlayerTrack.Quest.quests[1, 8].coord_found = P.currentcoord
            exitActionLoop()()
        if (P.PlayerTrack.Quest.quests[1, 8].status == 'started') and ('fruit' in P.items):
            exc.pop('Wild Herd')
            exc['Baby Mammoth'] = [2, 6, FindBabyMammoth]
        actions = {'Excavate':Excavate(exc,9)}
        actionGrid(actions, True)

def A_pond(inspect=False):
    exc = {'Go Fishing':[1,5,Gather('raw fish',0)],
           'Clay':[6,8,getItem('clay')],
           'Giant Serpent':[9,9,partial(encounter,'Giant Serpent',[20,45],['Elemental'],{'scales':[1,3]})]}
    if inspect:
        return exc, 12
    else:
        P = lclPlayer()
        if (P.PlayerTrack.Quest.quests[2, 4].status == 'started') and (P.currentcoord == P.PlayerTrack.Quest.quests[2, 4].pond):
            # The player encounters Serpent Mother instead
            def Reward(_=None):
                # Companion leaves and player gets scales and raw fish
                P.group.pop("Companion")
                P.PlayerTrack.Quest.update_quest_status((2, 4), 'complete')
                P.parentBoard.gridtiles[P.PlayerTrack.Quest.quests[2, 4].pond].color = (1,1,1,1)
                P.addItem('scales', 3)
                P.addItem('raw fish', 2)
            exc.pop('Giant Serpent')
            exc['Mother Serpent'] = [9,9,partial(encounter,'Mother Serpent',[50,50],['Elemental'],Reward)]
        elif (P.PlayerTrack.Quest.quests[2, 4].status == 'complete'):
            # 2/3 chance that you get 2 scales without fighting
            if rbtwn(1, 3) <= 2:
                def func(_=None):
                    output("Found 2 scales instead of serpent", 'yellow')
                    P.useSkill("Excavating")
                    P.addItem('scales', 2)
                    exitActionLoop()()
            else:
                func = partial(encounter,'Giant Serpent',[20,45],['Elemental'],{'scales':[1,3]})
            exc['Giant Serpent'][2] = func
        actions = {'Excavate':Excavate(exc,12)}
        actionGrid(actions, True)

def A_cave(tier=1, inspect=False):
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
    if inspect:
        return exc, 20
    else:
        action_tiers = {1:{'Excavate':Excavate(exc[1],20),'Descend':move_down},
                        2:{'Excavate':Excavate(exc[2],20),'Descend':move_down,'Ascend':move_up},
                        3:{'Excavate':Excavate(exc[3],20),'Ascend':move_up}}
        actionGrid(action_tiers[tier], True)

def A_outpost(inspect=False):
    exc = {'String':[1,3,getItem('string')],
           'Beads':[4,6,getItem('beads')],
           'Sand':[7,8,getItem('sand')],
           'Bandit':[9,10,C_outpost(True)]}
    if inspect:
        return exc, 12
    else:
        actions = {'Excavate':Excavate(exc,12)}
        actionGrid(actions, True)

def A_mountain(tier=1, inspect=False):
    def move_down(_):
        P = lclPlayer()
        if P.paused:
            return
        C_mountain(tier-1)
    def move_up(_):
        P = lclPlayer()
        if P.paused:
            return
        if (P.PlayerTrack.Quest.quests[3, 2].status=='started') and ((tier+1) == 3):
            P.PlayerTrack.Quest.quests[3, 2].reached_top = True
            if (P.PlayerTrack.Quest.quests[3, 2].total_ore_cost >= 5):
                P.PlayerTrack.Quest.quests[3, 2].completed_mountain = True
                output("Now go to the plains and find at least 5 pieces of meat!", 'blue')
        C_mountain(tier+1)
    exc = {1:{'Copper':[1,5,getItem('copper')],'Iron':[6,10,getItem('iron')],'Monk':[11,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           2:{'Kevlium':[1,4,getItem('kevlium')],'Nickel':[5,8,getItem('nickel')],'Monk':[9,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           3:{'Diamond':[1,3,getItem('tungsten')],'Chromium':[4,6,getItem('titanium')],'Monk':[7,8,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]}}
    if inspect:
        return exc, 20
    else:
        P = lclPlayer()
        if (P.PlayerTrack.Quest.quests[3, 3].status == 'started') and (not hasattr(P.PlayerTrack.Quest.quests[3, 3], 'convinced_monk')):
            def convince_monk(_=None):
                persuasion = P.activateSkill("Persuasion")
                r = rbtwn(1, 12, None, persuasion, 'Persuasion ')
                if r <= persuasion:
                    P.useSkill("Persuasion")
                    output("You were able to convince the monk to come with you. Study with him at your city.", 'blue')
                    P.PlayerTrack.Quest.quests[3, 3].convinced_monk = 0
                else:
                    output("You were unable to convince the monk to come with you." , 'yellow')
                exitActionLoop()()
            for i in range(1, 4):
                exc[i]['Monk'][2] = convince_monk
        action_tiers = {1:{'Excavate':Excavate(exc[1],20),'Ascend':move_up},
                        2:{'Excavate':Excavate(exc[2],20),'Descend':move_down,'Ascend':move_up},
                        3:{'Excavate':Excavate(exc[3],20),'Descend':move_down}}
        actionGrid(action_tiers[tier], True)

def A_oldlibrary(inspect=False):
    exc = {'Book':[1,8,getBook()],'Hermit':[9,14,C_oldlibrary(True)]}
    if inspect:
        return exc, 20
    else:
        actions = {'Excavate':Excavate(exc,20)}
        actionGrid(actions, True)

def A_ruins(inspect=False):
    def read_attempt(_):
        P = lclPlayer()
        book = getBook(True)()
        skill = book2skill(book)
        if P.skills[skill] < 8:
            critthink = P.activateSkill('Critical Thinking')
            r = rbtwn(1, 12, None, critthink, 'Critical Thinking ')
            if r <= critthink:
                output(f"Understood teachings from old {book}.",'green')
                P.updateSkill(skill, 1, 8)
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
    if inspect:
        return exc, 20
    else:
        P = lclPlayer()
        if (P.PlayerTrack.Quest.quests[4, 6].status == 'started'):
            exc['Old Book'][2] = StoreTatteredBook
        actions = {'Excavate':Excavate(exc,20)}
        actionGrid(actions, True)

def A_battlezone(inspect=False):
    def rare_find(_):
        r = rbtwn(1, 4)
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
    if inspect:
        return exc, 20
    else:
        actions = {'Excavate':Excavate(exc,20)}
        actionGrid(actions, True)

def A_wilderness(inspect=False):
    P = lclPlayer()
    def rare_find(_):
        r = rbtwn(1, 6)
        if (P.PlayerTrack.Quest.quests[5, 1].status == 'started') and (not P.PlayerTrack.Quest.quests[5, 1].has_gold) and (r <= 4):
            output("You found gold! Go back to the your city and get it smithed!", 'blue')
            P.PlayerTrack.Quest.quests[5, 1].has_gold = True
            exitActionLoop(empty_tile=True)()
        elif r==1:
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
        gathering = P.activateSkill("Gathering")
        if gathering > 0:
            P.useSkill("Gathering", 2, 7)
            item = 'fruit' if rbtwn(1,2)==1 else 'bark'
            output(f"You gather {gathering} {item}")
            getItem(item, gathering, empty_tile=True)
        else:
            exitActionLoop(empty_tile=True)()
    exc = {'Rare Find':[1,3,rare_find],'Grand Tree':[4,10,tree_gather],'Nothing':[11,20,exitActionLoop(empty_tile=True)]}
    if inspect:
        return exc, 20
    else:
        actions = {'Excavate':Excavate(exc,20)}
        actionGrid(actions, True)

avail_actions = {'road':A_road,'plains':A_plains,'pond':A_pond,'cave':A_cave,'outpost':A_outpost,'mountain':A_mountain,'oldlibrary':A_oldlibrary,'ruins':A_ruins,'battle1':A_battlezone,'battle2':A_battlezone}

#%% Game Properties
food_restore = var.food_restore
ore_properties = var.ore_properties
capital_info = var.capital_info
city_info = var.city_info
village_invest = var.village_invest
connectivity = var.connectivity

def conn2set():
    P = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, i:])) for i in range(len(connectivity))]
    cityOrder = sorted(capital_info)
    S, T = {}, {}
    for i in range(len(P)):
        t = set()
        for j in range(len(P[i])):
            if P[i][j] > 0:
                S[frozenset([cityOrder[i], cityOrder[j]])] = P[i][j]
                t.add(cityOrder[j])
        T[cityOrder[i]] = t # Tension Cities
    return [S, T]
Skirmishes = [{}] + conn2set()

# Get the inverse of ore_properties
def get_inverse_ore_properties():
    ivore = {cmbtstyle: [None]*4 for cmbtstyle in ['Elemental','Physical','Trooper','Wizard']}
    for ore in ore_properties:
        ivore[ore_properties[ore][0]][ore_properties[ore][1]-1] = ore
    return ivore
inverse_ore_properties = get_inverse_ore_properties()
# Get the inverse of city_info
def get_inverse_city_info():
    adept, master = {}, {}
    for city, D in city_info.items():
        for ability, lvl in D.items():
            if ability in {'sell', 'entry'}:
                continue
            if lvl >= 8:
                if ability in adept:
                    adept[ability].append(city)
                else:
                    adept[ability] = [city]
            if lvl >= 12:
                if ability in master:
                    master[ability].append(city)
                else:
                    master[ability] = [city]
    master['Hit Points'] = ['ruins']
    master['Agility'].append('plains')
    master['Cunning'].append('old libraries')
    master['Critical Thinking'] = ['old libraries']
    master['Heating'] = ['ruins']
    master['Survival'] = ['mountains']
    master['Gathering'] = ['plains']
    master['Excavating'].append('mountains')
    return adept, master
adept_loc, master_loc = get_inverse_city_info()

#%% Player
class Player(Image):
    def __init__(self, board, username, birthcity, **kwargs):
        super().__init__(**kwargs)
        self.source = f'images\\characters\\{birthcity}.png'
        
        # Game Properties
        self.parentBoard = board
        self.username = username
        self.birthcity = birthcity
        self.is_trading = False
        self.imgSize = pilImage.open(self.source).size
        self.rbtwn = rbtwn
        self.output = output
        self.exitActionLoop = exitActionLoop

        # Player Victory Points
        self.Combat = 3
        self.Capital = 0
        self.Reputation = 0
        self.Knowledge = 0
        self.Fellowship = 0
        self.Titles = 0
        self.TotalVP = 3

        # Constraints
        self.round_ended = False
        self.paused = False
        self.started_round = False
        self.ate = 0
        self.max_eating = 2
        self.max_road_moves = 2
        self.max_actions = 2
        self.max_minor_actions = 3
        self.max_capacity = 3
        self.max_fatigue = 12
        self.stability_impact = 1
        self.combatstyle = cities[self.birthcity]['Combat Style']

        # Current Stats
        self.road_moves = 2
        self.actions = 2
        self.minor_actions = 3
        self.item_count = 0
        self.items = {}
        self.activated_bartering = False
        self.first_purchase_after_barter = False
        self.bartering_mode = 0
        self.unsellable = set()
        self.fatigue = 0
        self.dueling_hiatus = 0
        self.player_fight_hiatus = {}
        self.free_smithing_rent = False
        self.city_discount = 0
        self.city_discount_threshold = float('inf')
        self.standard_read_xp = 4
        self.group = {}
        self.has_warrior = 0
        self.training_discount = False
        self.round_num = 0

        self.coins = cities[self.birthcity]['Coins']
        self.working = [None, 0]
        self.paralyzed_rounds = 0
        self.tiered = False
        self.cspace = [[None]*7, [None]*7]
        self.aspace = [[None, None], [None]*4, [None]*4]

        # Player Track
        #Combat
        self.combatxp = 0
        self.attributes = {'Agility':0,'Cunning':1,'Technique':2,'Hit Points':3,'Attack':4,'Stability':5,'Def-Physical':6,'Def-Wizard':7, 'Def-Elemental':8, 'Def-Trooper':9}
        self.atrorder = ['Agility','Cunning','Technique','Hit Points','Attack','Stability','Def-Physical','Def-Wizard','Def-Elemental','Def-Trooper']
        self.combat = np.array([0, 0, 0, 2, 1, 0, 0, 0, 0, 0])
        self.boosts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.perm_boosts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # permanent boosts will enable fellowship joining
        for atr, val in cities[self.birthcity]['Combat Boosts']:
            self.boosts[self.attributes[atr]] += val
            self.perm_boosts[self.attributes[atr]] += val
        self.current = self.combat + self.boosts
        #Knowledge
        self.skills = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
        self.xps = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
        for skl, val in cities[self.birthcity]['Knowledges']:
            self.updateSkill(skl, val)
        # Combination of Abilities for training
        self.trained_abilities = {**self.skills, **self.attributes} # This syntax requires Python 3.5 or greater
        for ability in self.trained_abilities:
            self.trained_abilities[ability] = False
        #Capital
        self.cityorder = sorted(cities)
        self.homes = {city: False for city in cities}
        self.markets = {city: False for city in cities}
        self.tended_market = {city: False for city in cities}
        self.workers = {city: False for city in cities}
        self.bank = {city: 0 for city in cities}
        self.villages = {city: {} for city in cities}
        self.awaiting = {city: {} for city in cities}
        # Some helpers of Capital
        self.already_asset_bartered = False
        self.training_allowed = {city: False for city in cities}
        self.training_allowed[self.birthcity] = True
        self.market_allowed = {city: False for city in cities}
        self.market_allowed[self.birthcity] = True
        # Reputation
        self.reputation = np.empty((6, 8), dtype=object)
        for stage in range(6):
            for mission in range(8):
                self.reputation[stage, mission] = {}
        self.entry_allowed = {city: False for city in cities}
        self.entry_allowed[self.birthcity] = True
        # Fellowships
        self.castle_of_conjurors = {'delay': 0, 'lvl': 0, 'rounds': 0, 'bond': 0, 'combat_limit': 1, 'combats': 0}  # anafola castle of conjurors
        self.grand_library = {'read_fatigue': 0, 'borrowed': {}, 'total_borrowed': 0, 'conseq_read_counter': 0, 'read_action_counter': 0, 'books_learned': 0}
        self.grand_bank = {'credit_score': 0, 'loan_amount': None, 'original_loan_length': None, 'loan_length_remaining': None, 'strikes': 0, 'total_strikes': 0, } # demetry grand central bank
        self.reaquisition_guild = {'class': 1, 'success': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}} # enfeir requisition guild
        self.defenders_guild = {}
        self.peace_embassy = {}
        self.ancestrial_order = {'loot_class': 0, 'loot_class_progress': 0, 'food_class': 0, 'food_class_progress': 0, 'fatigue_class': 0, 'fatigue_class_progress': 0, 'minor_treading': False, 'minor_treading_progress': 0, 'horse_riding': False, 'horse_riding_progress': 0, 'major_treading': False, 'major_treading_progress': 0, 'first_class_1': None, 'first_class_2': None, 'first_class_3': None} # kubani ancestrial order
        self.meditation_chamber = {'class': 1, 'score': 0, 'success': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}} # pafiz meditation chamber
        self.colosseum = {}
        self.ancient_magic_museum = {'gifted_museum': 0} # starfex ancient magic museum
        self.smithing_guild = {}
        self.wizard_tower = {'membership': None, 'membership_rounds_remaining': None, 'queued_membership': None, 'auto_renewal': 'off', 'renew_with_market_earnings': 'off', 'current_consecutive_platinum_renewals': 0, 'best_consecutive_platinum_renewals': 0} # Tamariza wizard tower
        self.hunters_guild = {}
        self.hidden_lair = {'sharpness': 0, 'invincibility': 0, 'vanish': 0, 'shadow': 0, 'vision': 0, 'sharpness_progress': 0, 'invincibility_progress': 0, 'vanish_progress': 0, 'shadow_progress': 0, 'vision_progress': 0} # zinzibar ninja lair
        self.fellowships = {city: getattr(hmrk, city)(self) for city in var.cities}
        
        # Fellowship Statuses
        self.teleport_ready = False
        self.has_horse = False
        self.loot_bonus = 0
        self.food_bonus = 0
        
        # Titles - note that minTitleReq has been deprecated in this dict in favor of using the server.
        self.titles = {'explorer': {'titleVP': 5, 'minTitleReq': 20, 'value':0, 'category': 'General', 'description': 'Most unique tiles traveled upon.'},
                      'loyal': {'titleVP': 2, 'minTitleReq': 25, 'value':0, 'category': 'General', 'in_birthcity':True, 'description': 'Most rounds spent in their birth city (all actions per round).'},
                      'valiant': {'titleVP': 3, 'minTitleReq': 6, 'value':0, 'category': 'Combat', 'description': 'Maximum difference between an opponent stronger than you, that you defeated, and your total combat at the start of battle.'},
                      'sorcerer': {'titleVP': 5, 'minTitleReq': 2, 'value':0, 'category': 'Fellowship', 'description': 'Most consecutive Tamariza Wizard Tower platinum membership renewals.'},
                      'superprime': {'titleVP': 5, 'minTitleReq': 40, 'value':0, 'category': 'Fellowship', 'description': 'Highest credit score from the Demetry Grand Bank.'},
                      'traveler': {'titleVP': 3, 'minTitleReq': 30, 'value':0, 'category': 'General', 'description': 'Most road tile movement.'},
                      'apprentice': {'titleVP': 4, 'minTitleReq': 60, 'value':0, 'category': 'Knowledge', 'description': 'Most cumulative levels gained from trainers.'},
                      'scholar': {'titleVP': 5, 'minTitleReq': 5, 'value':0, 'category': 'Fellowship', 'description': 'Most books checked out from the Benefriege public library, learned from, and returned.'},
                      'laborer': {'titleVP': 3, 'minTitleReq': 10, 'value':0, 'category': 'Capital', 'description': 'Most jobs worked.'},
                      'valuable': {'titleVP': 2, 'minTitleReq': 20, 'value':0, 'category': 'Capital', 'description': 'Most money earned from jobs worked.'},
                      'entrepreneur': {'titleVP': 3, 'minTitleReq': 50, 'value':0, 'category': 'Capital', 'description': 'Most money received from owned markets.'},
                      'trader': {'titleVP': 4, 'minTitleReq': 5, 'value':0, 'category': 'Capital', 'unique_traders': set(), 'description': 'Most trading (buy or sell) with unique traders.'},
                      'negotiator': {'titleVP': 4, 'minTitleReq': 15, 'value':0, 'category': 'Knowledge', 'description': 'Most successful persuasions.'},
                      'steady': {'titleVP': 4, 'minTitleReq': 15, 'value':0, 'category': 'General', 'description': 'Most rounds started with zero fatigue.'},
                      'grinder': {'titleVP': 3, 'minTitleReq': 120, 'value':0, 'category': 'General', 'description': 'Most fatigue taken.'},
                      'merchant': {'titleVP': 2, 'minTitleReq': 40, 'value':0, 'category': 'Capital', 'description': 'Most coin made from selling items to merchants or traders.'},
                      'brave': {'titleVP': 5, 'minTitleReq': 80, 'value':0, 'category': 'Combat', 'currentStreak': 0, 'description': 'Sum total level of continuous enemies defeated without restoring HP.'},
                      'decisive': {'titleVP': 1, 'minTitleReq': -float('inf'), 'category': 'General', 'value':None, 'sum': 0, 'round': 0, 'startTime': time(), 'description': 'Shortest average round time.'},
                      'champion': {'titleVP': 7, 'minTitleReq': 1, 'value':0, 'category': 'Fellowship', 'description': 'Most class V Scetcher Tournament titles won.'},
                      'resurrector': {'titleVP': 10, 'minTitleReq': 1, 'value':0, 'category': 'End Game', 'description': 'Strongest magic sword revived with its ability unlocked.'}}
        for key in self.titles:
            self.titles[key]['maxRecord'] = {'value': -float('inf') if key=='decisive' else 0, 'holder': None, 'title': key}
        self.titleOrder = [T[0] for T in sorted(self.titles.items(), key=lambda kv: kv[1]['titleVP'])]
        self.titleIndex = {T: i for i, T in enumerate(self.titleOrder)}

        # Position Player
        self.currentcoord = positions[birthcity][0]
        self.currenttile = self.parentBoard.gridtiles[self.currentcoord]
        self.moveto(self.currentcoord, False, True)
        self.size_hint = (self.imgSize[0]/xsize, self.imgSize[1]/ysize)
    def updateTitleValue(self, title, value=1, set_value=None):
        if title == 'decisive':
            # value is ignored in this caupdse
            endTime = time()
            self.titles[title]['sum'] += (endTime - self.titles[title]['startTime'])
            self.titles[title]['round'] += 1
            self.titles[title]['value'] =  - (self.titles[title]['sum'] / self.titles[title]['round']) # Force negative for consistency in finding max holder
        else:
            if set_value is not None:
                self.title[title]['value'] = set_value
            else:
                self.titles[title]['value'] += value
        socket_client.send('[TITLE VALUE]', {'value': self.titles[title]['value'], 'username': self.username, 'title': title})
        if hasattr(self, 'PlayerTrack'):
            self.PlayerTrack.update_single_title(title)
    def newMaxRecord(self, maxRecord):
        title = maxRecord['title']
        previousRecord = self.titles[title]['maxRecord']
        if (previousRecord['holder'] == self.username) and (maxRecord['holder'] != self.username):
            # If this user was the last holder, then remove Title VP
            self.Titles -= self.titles[title]['titleVP']
            self.updateTotalVP(-self.titles[title]['titleVP'], False)
            output(f"You lost The {title.capitalize()} title to {maxRecord['holder']}", 'red')
        elif (previousRecord['holder'] != self.username) and (maxRecord['holder'] == self.username):
            # This means he is the new holder of the title!
            output(f"You now hold The {title.capitalize()} title!", 'green')
            self.titles[title]['maxRecord'] = {'value': self.titles[title]['value'], 'holder': self.username, 'title': title}
            self.Titles += self.titles[title]['titleVP']
            self.updateTotalVP(self.titles[title]['titleVP'], False)
        elif (previousRecord['holder'] != self.username) and (maxRecord['holder'] != self.username) and (maxRecord['holder'] != previousRecord['holder']):
            output(f"{maxRecord['holder']} now holds The {maxRecord['title'].capitalize()} title!", 'blue')
        self.titles[maxRecord['title']]['maxRecord'] = maxRecord
        if hasattr(self, 'PlayerTrack'):
            self.PlayerTrack.update_single_title(title)
    def savePlayer(self):
        self.PlayerTrack.Quest.saveTable()
        self.PlayerTrack.craftingTable.saveTable()
        self.PlayerTrack.armoryTable.saveTable()
        exclusions = {'_coreimage','_loops','_context','_disabled_value','_disabled_count','canvas','_proxy_ref','_loop','parentBoard','currenttile', 'mtable', 'PlayerTrack', 'fellowships', 'output', 'exitActionLoop', 'rbtwn'}
        myVars = {field: self.__dict__[field] for field in set(self.__dict__).difference(exclusions)}
        myVars['output messages'] = self.parentBoard.game_page.output.messages[1:]
        myVars['seed'] = seed
        myVars['gameEnd'] = game_app.launch_page.end_lbl.text.split('\n')[1]
        myVars['capital info'] = capital_info
        myVars['skirmishes'] = [Skirmishes[1], Skirmishes[0]]
        if not os.path.exists(f'saves\\{self.username}'):
            os.makedirs(f'saves\\{self.username}')
        with open(f'saves\\{self.username}\\{save_file}.pickle', 'wb') as f:
            pickle.dump(myVars, f)
    def loadPlayer(self):
        global Skirmishes, capital_info
        with open(f'saves\\{self.username}\\{load_file}.pickle', 'rb') as f:
            myVars = pickle.load(f)
        for field, value in myVars.items():
            if field in {'username', 'output messages', 'seed', 'gameEnd', 'capital_info', 'skirmishes'}:
                continue
            setattr(self, field, value)
        if 'output messages' in myVars:
            for msg in myVars['output messages']:
                self.parentBoard.game_page.output.add_msg(msg)
        capital_info = myVars['capital info']
        Skirmishes[0] = myVars['skirmishes'][1] # They are opposites in the server.
        Skirmishes[1] = myVars['skirmishes'][0]
        self.PlayerTrack.Quest.loadTable()
        self.PlayerTrack.craftingTable.loadTable()
        self.PlayerTrack.armoryTable.loadTable()
        self.fellowships = {city: getattr(hmrk, city)(self) for city in var.cities}
        self.moveto(self.currentcoord, False, True)
        socket_client.send('[MOVE]', self.currentcoord) # So that everyone else can see the move.
        self.parentBoard.startRound()
    def updateView(self):
        self.size_hint = (self.imgSize[0]*(self.parentBoard.zoom+1)/xsize, self.imgSize[1]*(self.parentBoard.zoom+1)/ysize)
        self.pos_hint = {'center_x':self.currenttile.centx, 'center_y':self.currenttile.centy}
    def get_recover(self):
        if (self.currenttile.tile in self.homes) and self.homes[self.currenttile.tile]:
            return 4
        elif self.currenttile.tile == self.birthcity:
            return 2
        return 1
    def go2consequence(self, amt=1):
        if (self.currenttile.tile in cities) and capital_info[self.currenttile.tile]['trader allowed']:
            if rbtwn(1, 8) == 1:
                self.currentile.trader_appears()
        if self.currenttile.tile in consequences:
            if self.tiered:
                # In the case that they are in the mountain or cave, then make sure they go to the right tier.
                consequences[self.currenttile.tile](self.tiered)
            else:
                consequences[self.currenttile.tile]()
        elif self.currenttile.tile in cities:
            city_consequence(self.currenttile.tile)
        else:
            exitActionLoop(None,amt)()
    def go2action(self, param=None):
        if (self.currenttile.tile in avail_actions) and (not self.currenttile.is_empty):
            if param is None:
                avail_actions[self.currenttile.tile]()
            else:
                avail_actions[self.currenttile.tile](param)
        elif self.currenttile.tile in cities:
            city_actions(self.currenttile.tile)
        else:
            actionGrid({}, True)
    def make_citypage(self, city):
        screen = Screen(name='City')
        self.parentBoard.game_page.city_page = CityPage(city, self)
        screen.add_widget(self.parentBoard.game_page.city_page)
        self.parentBoard.game_page.main_screen.add_widget(screen)
        self.parentBoard.game_page.main_screen.current = 'City'
        self.parentBoard.game_page.cityscreen = screen
    def remove_citypage(self):
        if self.parentBoard.game_page.city_page is not None:
            self.parentBoard.game_page.main_screen.current = "Board" if self.parentBoard.game_page.toggleView.text=="Player Track" else "Player Track"
            self.parentBoard.game_page.main_screen.remove_widget(self.parentBoard.game_page.cityscreen)
            self.parentBoard.game_page.city_page = None
    def moveto(self, coord, trigger_consequence=True, skip_check=False):
        if hasattr(self, 'PlayerTrack'):
            if self.PlayerTrack.Quest.quests[3, 6].status == 'started':
                output("You cannot move while fighting in Skirmish!", 'yellow')
                return
        nxt = self.parentBoard.gridtiles[coord]
        if nxt.tile in cities:
            # Can't enter the city unless was fainted and rushed to the hospital"
            if (not self.entry_allowed[nxt.tile]) and (self.current[self.attributes["Hit Points"]] > 0):
                output(f"The city will not let you enter! (Need {city_info[nxt.tile]['entry']}+ Reputation!)",'red')
                return
        if (not skip_check) and (self.currenttile.tile != 'road') and (self.currenttile.tile not in cities) and (nxt.tile=='road'):
            output("Move there on same action (+2 Fatigue)?", 'blue')
            actionGrid({'Yes':(lambda _:self.moveto(coord,trigger_consequence,3)),'No':(lambda _:self.moveto(coord,trigger_consequence,2))}, False, False)
        else:
            self.remove_citypage() # go back to board page before the avatar moves
            if skip_check != True:
                socket_client.send('[MOVE]',coord)
            self.currenttile.source = f'images\\tile\\{self.currenttile.tile}.png' # remove green outline of tile moving away from.
            self.currentcoord = coord
            self.currenttile = self.parentBoard.gridtiles[coord]
            if not self.currenttile.traveledOn:
                self.updateTitleValue('explorer')
                self.currenttile.traveledOn = True
            if self.currenttile.tile != self.birthcity:
                self.titles['loyal']['in_birthcity'] = False
            if self.currenttile.tile == 'road':
                self.updateTitleValue('traveler', 1)
            self.currenttile.source = f'images\\ontile\\{self.currenttile.tile}.png' # outline new tile with green border.
            # Quest completions/consequences (if any)
            if hasattr(self, 'PlayerTrack'):
                if (self.PlayerTrack.Quest.quests[2, 3].status == 'started') and (self.PlayerTrack.Quest.quests[2, 3].furthest_city==self.currenttile.tile):
                    # The player made it to the furthest city without failing, give reward
                    self.PlayerTrack.Quest.update_quest_status((2, 3), 'complete')
                    output("The nobleman gives you 10 coins for your service", 'green')
                    self.coins += 10
                elif (self.PlayerTrack.Quest.quests[2, 6].status == 'started') and (self.PlayerTrack.Quest.quests[2, 6].furthest_city==self.currentile.tile):
                    self.PlayerTrack.Quest.update_quest_status((2, 6), 'complete')
                    output("You now have one additional move on the road!", 'green')
                    self.max_road_moves += 1
                elif (self.PlayerTrack.Quest.quests[3, 1].status == 'complete') and (self.currenttile.tile==self.birthcity):
                    # Moving into your birth city after completing 'Feed the Poor' quest has a chance of being given some well cooked food
                    if rbtwn(1, 4) == 1:
                        food = ['well cooked fish', 'well cooked meat'][rbtwn(0,1)]
                        output(f"City villagers gift you a {food}!", 'green')
                        self.addItem(food)
                elif (self.PlayerTrack.Quest.quests[3, 2].status == 'started') and (self.currenttile.tile in cities):
                    resetFitnessTraining()

            if (self.currenttile.tile in cities):
                # Make City Page
                self.make_citypage(self.currenttile.tile)
                # Collect any money waiting at bank
                self.coins += self.bank[self.currenttile.tile]
                self.updateTitleValue('entrepreneur', self.bank[self.currenttile.tile])
                self.bank[self.currenttile.tile] = 0
                if hasattr(self, 'PlayerTrack'):
                    self.PlayerTrack.craftingTable.enable_selling()
                    self.PlayerTrack.armoryTable.enable_selling()
            elif (self.currenttile.trader_rounds > 0) and hasattr(self, 'PlayerTrack'):
                self.PlayerTrack.craftingTable.enable_selling()
                self.PlayerTrack.armoryTable.enable_selling()
            elif hasattr(self, 'PlayerTrack'):
                self.PlayerTrack.craftingTable.disable_selling()
                self.PlayerTrack.armoryTable.disable_selling()
            # Collect any awaiting investments (if moved to a village)
            self.getAwaiting()
            game_app.game_page.recover_button.text = f'Rest ({self.get_recover()})'
            self.parentBoard.updateView()
            self.pos_hint = {'center_x':self.currenttile.centx, 'center_y':self.currenttile.centy}
            # Take damage and action afterwards
            if skip_check == 3:
                self.takeDamage(0,2)
            elif skip_check == 2:
                output(f"[ACTION {self.max_actions-self.actions+1}] Moving onto road consumes action.")
                self.actions -= 1
                self.takeDamage(0,1)
            if trigger_consequence:
                self.go2consequence()
        if self != self.parentBoard.localPlayer:
            # Check for player interactions
            self.parentBoard.game_page.check_players()
    def add_PlayerTrack(self):
        self.PlayerTrack = PlayerTrack(self)
        screen = Screen(name='Player Track')
        screen.add_widget(self.PlayerTrack)
        self.parentBoard.game_page.main_screen.add_widget(screen)
    def get_mainStatUpdate(self):
        return [f'{self.fatigue}/{self.max_fatigue}',
                f'{self.current[self.attributes["Hit Points"]]}/{self.combat[self.attributes["Hit Points"]]+self.boosts[self.attributes["Hit Points"]]}',
                f'{self.coins}',
                f'{self.item_count}/{self.max_capacity}',
                f'{self.actions}|{self.minor_actions}|{self.road_moves}|{self.max_eating-self.ate}']
    def add_mainStatPage(self):
        self.mtable = Table(['Fatigue','HP','Coins','Items','M|m|r|e'],[self.get_mainStatUpdate()],False,(1,1,1,0.5),text_color=(0,0,0,0.9),
                            pos_hint={'x':0,'y':game_app.game_page.stat_ypos}, size_hint_y=game_app.game_page.stat_ysize)
        game_app.game_page.right_line.add_widget(self.mtable)
    def update_frontStats(self):
        updated_data = self.get_mainStatUpdate()
        for i in range(len(updated_data)):
            self.mtable.cells[self.mtable.header[i]][0].text = updated_data[i]
    def update_mainStatPage(self):
        self.update_frontStats()
        self.PlayerTrack.updateAll()
        self.PlayerTrack.Quest.update_reqs()
    def recover(self, rest_rate=None, _=None):
        if not self.paused:
            rest_rate = rest_rate if type(rest_rate) is int else self.get_recover()
            if self.activated_bartering:
                # If bartered, account for the fatigue here and then specify in takeAction to overlook
                output("Take an extra fatigue for bartering this turn", 'yellow')
                self.fatigue += 1
            self.fatigue = max([0, self.fatigue-rest_rate])
            hp_idx = self.attributes['Hit Points']
            prior_hp = self.current[hp_idx]
            self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+rest_rate])
            if prior_hp != self.current[hp_idx]:
                # If resting increases hp, then reset current streak.
                self.titles['brave']['currentStreak'] = 0
            output(f'[ACTION {self.max_actions-self.actions+1}] You rested {rest_rate} fatigue/HP')
            self.takeAction(0, False, True)
    def attempt_barter(self, following_function, _=None):
        if self.paused:
            return
        if self.activated_bartering:
            following_function()
        self.activated_bartering = True
        bartering = self.activateSkill('Bartering')
        r = rbtwn(1, 12, None, bartering, 'Bartering ')
        if r <= bartering:
            self.useSkill('Bartering')
            output("You successfully Barter", 'green')
            self.bartering_mode = 2 if bartering > 8 else 1
            self.first_purchase_after_barter = True
            following_function()
        else:
            output("You fail to Barter", 'yellow')
            self.bartering_mode = 0
            following_function() # Not given the opportunity to barter again unless performs a different action first
    def eat(self, food, _=None):
        if self.paused:
            return
        if self.ate >= self.max_eating:
            output("You can't eat anymore this action!",'yellow')
            return
        self.ate += 1
        ftg, hp = food_restore[food]
        if self.player.food_bonus > 0:
            if ftg > 0:
                ftg += self.player.food_bonus
            if hp > 0:
                hp += self.player.food_bonus
        survival = self.skills["Survival"]
        r = rbtwn(1, 12, None, survival, 'Survival ')
        if r <= survival:
            if rbtwn(0,1):
                output("Your Survival skill restored +1HP!",'green')
                hp += 1
            else:
                output("Your Survival skill restored +1FTG!",'green')
                ftg += 1
            self.useSkill("Survival")
        restoring_list = []
        if ftg > 0: restoring_list.append(f'FTG by {ftg}')
        if hp > 0:
            restoring_list.append(f'HP by {hp}')
            self.titles['brave']['currentStreak'] = 0
        output(f'Restored {", ".join(restoring_list)}','green')
        self.addItem(food, -1)
        self.fatigue = max([0, self.fatigue - ftg])
        hp_idx = self.attributes['Hit Points']
        self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+hp])
        self.update_mainStatPage()
    def end_round(self):
        self.paused = True
        self.round_ended = True
        output(f"{self.username} Ended Round")
        print(self.parentBoard.Players, game_app.launch_page.usernames)
        # Update local end round stats:
        if self.titles['loyal']['in_birthcity']:
            self.updateTitleValue('loyal', 1)
        self.updateTitleValue('decisive')
        self.fellowships['demetry'].end_round()
        self.fellowships['tamariza'].end_round()
        if len(self.parentBoard.Players) != len(game_app.launch_page.usernames):
            # If not all players have been created then do not end the round yet
            return
        for P in self.parentBoard.Players.values():
            print(P.username, P.round_ended)
            if not P.round_ended:
                # If someone has not ended the round yet, then wait.
                return
        # Otherwise start the round
        self.parentBoard.startRoundDelay()
    def takeDamage(self, hp_amt, ftg_amt, add=True):
        hp_idx = self.attributes['Hit Points']
        self.current[hp_idx] = max([0, self.current[hp_idx]-hp_amt])
        fainted = False
        if self.current[hp_idx] == 0:
            self.parentBoard.sendFaceMessage('You Fainted!','red')
            if self.has_horse and (self.currenttile.tile not in var.cities):
                self.has_horse = False
                self.max_road_moves -= 3
                output("You lost your horse!", 'red')
                self.update_frontStats()
            if (self.Combat > 2):
                survival = self.activateSkill("Survival")
                r = rbtwn(1, 12, None, survival, 'Survival ')
                # Survival is activated and may save you from losing attributes
                if r <= survival:
                    output("Your survival skill enabled you to save your attributes!")
                else:
                    while True:
                        Ratr = np.random.choice(self.atrorder)
                        if ((Ratr=='Attack') or (Ratr=='Hit Points')):
                            if (self.combat[self.attributes[Ratr]]>1):
                                break
                        elif self.combat[self.attributes[Ratr]] > 0:
                            break
                    self.updateAttribute(Ratr, -1)
                    output(f"You lose {Ratr}, are paralyzed, and rushed to nearest city.",'red')
            # Just in case they die in a tiered environment, make sure they are not stuck.
            self.tiered = False
            # Move to nearest city
            coord, distance, path = self.currenttile.findNearest(cities)
            self.moveto(coord, trigger_consequence=False)
            # Artificially paralyze the player by setting fatigue 1 greater than max
            self.fatigue = self.max_fatigue + 1
            fainted = True
            # Check if Fainting fails/restarts any missions:
            if (self.PlayerTrack.Quest.quests[2, 3].status == 'started'):
                self.PlayerTrack.Quest.update_quest_status((2, 3), 'failed')
            if (self.PlayerTrack.Quest.quests[2, 6].status == 'started'):
                self.PlayerTrack.Quest.update_quest_status((2, 6), 'not started')
            if (self.PlayerTrack.Quest.quests[3, 2].status == 'started'):
                resetFitnessTraining()
            if (self.PlayerTrack.Quest.quests[3, 6].status == 'started'):
                self.PlayerTrack.Quest.update_quest_status((3, 6), 'failed')
        if not fainted:
            self.fatigue = self.fatigue + ftg_amt if add else ftg_amt
        self.update_mainStatPage()
        paralyzed = True if self.fatigue>self.max_fatigue else False
        check_paralysis()
        return (fainted or paralyzed)
    def takeAction(self, fatigue=1, verbose=True, from_recovery=False):
        if verbose: output(f"[ACTION {self.max_actions-self.actions+1}] You may take {fatigue} fatigue:")
        self.actions -= 1
        if self.activated_bartering:
            # Player performed an action after bartering - make sure you give them an extra fatigue if they bartered and didn't do 3 transactions except if rested
            if not from_recovery:
                output("Take an extra fatigue for bartering this turn", 'yellow')
                fatigue += 1
            self.activated_bartering = False
            self.first_purchase_after_barter = False
            self.bartering_mode = 0
        if (fatigue > 0) and (self.paralyzed_rounds == 0):
            # Not activated, but can be used.
            survival = self.skills["Survival"]
            r = rbtwn(1, 18, None, survival, 'Survival ') # At level 9 expect every 2nd action to not consume fatigue.
            if r <= survival:
                self.useSkill("Survival")
                fatigue -= 1
                output("Your survival skill saved you a fatigue!", 'green')
            if fatigue > 0:
                output("Took normal fatigue this action")
                self.takeDamage(0,fatigue)
                self.updateTitleValue('grinder', fatigue)
        self.minor_actions = self.max_minor_actions # Refresh minor actions
        self.ate = 0 # Refresh how much one can eat
        self.road_moves = self.max_road_moves # If action is taken, then the road moves should be refreshed.
        self.already_asset_bartered = False # If an action is taken, you can try to barter for the asset again.
        # Quest Adjustments
        if self.PlayerTrack.Quest.quests[3, 6].status == 'started':
            self.PlayerTrack.Quest.quests[3, 6].action += 1
            if self.PlayerTrack.Quest.quests[3, 6].action >= 6:
                self.PlayerTrack.Quest.update_quest_status((3, 6), 'complete')
                self.coins += 15
        # Hallmark Adjustments
        if self.grand_library['read_action_counter'] > 0:
            self.fellowships['benfriege'].update_field('read_action_counter', '-1')
            if self.grand_library['read_action_counter'] == 0:
                self.grand_library['conseq_read_counter'] = 0
                self.fellowships['benfriege'].update_field('read_fatigue', 0)
        self.update_mainStatPage()
        if self.actions <= 0:
            self.end_round()
            self.getIncome() # Get income before sending end of round to all players.
            socket_client.send("[ROUND]",'end')
        elif (self.PlayerTrack.Quest.quests[3, 6].status == 'started') and (self.PlayerTrack.Quest.quests[3, 6].action % 2):
            JoinFight()
        elif self.working[0] is not None:
            perform_labor()
    def updateTotalVP(self, add, checkGameEnd=True):
        self.TotalVP += add
        if game_app.game_page.vpView.text != 'VP Hidden':
            game_app.game_page.vpView.text = f'[color={game_app.game_page.hclr}]{self.TotalVP}[/color] VP'
        if (add > 0) and checkGameEnd:
            self.checkGameEnd()
    def updateFellowshipVP(self, add):
        self.Fellowship += add
        self.PlayerTrack.fellowshipTab.text = self.PlayerTrack.get_tab_text('Fellowship')
        self.updateTotalVP(add)
    def GameEnd(self, player_stats):
        best_total, best_users = -1, []
        for username, stats in player_stats.items():
            total = int(np.sum([stats[s] for s in stats]))
            vals = [f'{s}: {stats[s]}' for s in stats]
            output(f"{username}| "+', '.join(vals)+f', Total: {total}', 'blue')
            if total > best_total:
                best_users = [username]
                best_total = total
            elif total == best_total:
                best_users.append(username)
        winner = ' AND '.join(best_users)
        clr = 'green' if self.username in best_users else 'red'
        self.parentBoard.sendFaceMessage(winner+' Wins the Game!', clr, 60)
    def checkGameEnd(self):
        trigger = True
        all_tracks = np.array([self.Combat, self.Reputation, self.Knowledge, self.Capital, self.Fellowship])
        if (5 in gameEnd) and np.any(all_tracks < gameEnd[5]):
            trigger = False
        if (4 in gameEnd) and (np.sum(all_tracks < gameEnd[4])<4):
            trigger = False
        if (3 in gameEnd) and (np.sum(all_tracks >= gameEnd[3])<3):
            trigger = False
        if (2 in gameEnd) and (np.sum(all_tracks >= gameEnd[2])<2):
            trigger = False
        if (1 in gameEnd) and np.all(all_tracks < gameEnd[1]):
            trigger = False
        for track in ['Reputation', 'Combat', 'Capital', 'Combat', 'Fellowship']:
            if (track in gameEnd) and (self.__dict__[track] < gameEnd[track]):
                trigger = False
        if trigger:
            output("You Triggered End Game!", 'blue')
            socket_client.send('[GAME END]', '')
            socket_client.send('[END STATS]', {'Combat':self.Combat, 'Reputation':self.Reputation, 'Capital':self.Capital, 'Knowledge':self.Knowledge, 'Fellowship':self.Fellowship, 'Titles':self.Titles})
    def get_bonus(self, amt):
        amt = amt + amt*self.loot_bonus
        if int(amt) < amt:
            remainder = amt - int(amt)
            if np.random.rand() <= remainder:
                amt = int(amt) + 1
            else:
                amt = int(amt)
        amt = int(amt)
        return amt
    def updateSkill(self, skill, val=1, max_lvl=12):
        lvl_gain = max([0, min([max_lvl - self.skills[skill], val])])
        if lvl_gain != val:
            output(f"Your level in {skill} was not able to level beyond {max_lvl} with this action", 'yellow')
        self.skills[skill] += lvl_gain
        self.Knowledge += lvl_gain
        self.updateTotalVP(lvl_gain)
        if (skill == 'Crafting') and (hasattr(self, 'PlayerTrack')):
            self.PlayerTrack.craftingTable.update_lbls()
    def addXP(self, skill, xp, msg=''):
        if self.skills[skill] >= 8:
            # Ensure that xp is not beyond the max level of 8
            return
        output(f"{msg}Gained {xp}xp for {skill}.",'green')
        self.xps[skill] += xp
        while (self.xps[skill] >= (3 + self.skills[skill])):
            self.xps[skill] -= (3 + self.skills[skill])
            # When gaining xp, you can't level up beyond level 8
            previousLevel = self.skills[skill]
            self.updateSkill(skill, 1, 8)
            if self.skills[skill] != previousLevel:
                # Ensure that we only display this message if the player is actually leveling up.
                # For example, adding 100xp from cheat command could throw this message mutliple times otherwise.
                output(f"Leveled up {skill} to {self.skills[skill]}!",'green')
            else:
                output(f'Cannot level {skill} beyond {previousLevel} by adding XP!', 'yellow')
    def activateSkill(self, skill, xp=1, max_lvl_xp=2):
        lvl = self.skills[skill]
        add_lvl = 0
        if (skill.lower() in {'persuasion', 'bartering'}) and (self.wizard_tower['membership'] is not None):
            if (self.wizard_tower['membership'] == 'Basic') and (rbtwn(1, 3) == 1):
                output(f"Your Basic Tamariza Tower Membership successfully activated 1 lvl higher {skill}.", 'green')
                add_lvl = 1
            elif (self.wizard_tower['membership'] == 'Gold') and (np.random.rand() <= 0.9):
                output(f"Your Gold membership increases your {skill} lvl by 1!", 'green')
                add_lvl = 1
            elif self.wizard_tower['membership'] == 'Platinum':
                # Must have Platinum membership
                add_lvl = rbtwn(1, 2)
                output(f"Your Platinum membership increases your {skill} lvl by {add_lvl}!", 'green')
        if rbtwn(1,self.max_fatigue) <= self.fatigue:
            output(f"Fatigue impacted your {skill} skill.",'yellow')
            actv_lvl = np.max([0,lvl-self.fatigue])
        else:
            actv_lvl = lvl
            if lvl <= max_lvl_xp: self.addXP(skill, xp, 'Activation: ')
        return actv_lvl + add_lvl
    def useSkill(self, skill, xp=1, max_lvl_xp=5):
        if skill == 'Persuasion':
            self.updateTitleValue('negotiator', 1)
        if self.trained_abilities[skill]:
            newlevel = self.levelup(skill, 1, 12)
            self.trained_abilities[skill] = False
            self.updateTitleValue('apprentice', newlevel)
        if self.skills[skill] <= max_lvl_xp:
            self.addXP(skill, xp, 'Successful: ')
    def updateAttribute(self, attribute, val=1, max_lvl=12):
        lvl_gain = max([0, min([max_lvl - self.combat[self.attributes[attribute]], val])])
        if lvl_gain != val:
            output(f"Your level in {attribute} was not able to level beyond {max_lvl} with this action", 'yellow')
        self.combat[self.attributes[attribute]] += lvl_gain
        self.current[self.attributes[attribute]] += lvl_gain
        self.Combat += lvl_gain
        self.updateTotalVP(lvl_gain)
        self.update_mainStatPage()
    def applyBoost(self, attribute, val=1, update_stats=True):
        index = self.attributes[attribute]
        self.boosts[index] += val
        self.current[index] += val
        if update_stats: self.update_mainStatPage()
    def levelup(self, ability, val=1, max_lvl=12):
        if ability in self.skills:
            self.updateSkill(ability, val, max_lvl)
        else:
            self.updateAttribute(ability, val, max_lvl)
        return self.get_level(ability)
    def get_level(self, ability):
        if ability in self.skills:
            return self.skills[ability]
        return self.combat[self.attributes[ability]]
    def purchase(self, asset, city, barter=None, _=None):
        if self.paused:
            return
        cost = var.capital_info[city][asset]
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
                self.updateTotalVP(capital_info[city]['home_cap'])
                output(f"You successfully purchased a home in {city}!", 'green')
                output(f"+{capital_info[city]['capacity']} inventory space.", 'green')
                output(f"+{capital_info[city]['home_cap']} Capital.", 'green')
                output(f"Training in {city} is now allowed.", 'green')
            else:
                if self.markets[city]:
                    output('You already own a market here!','yellow')
                    exitActionLoop(amt=0)()
                    return
                self.markets[city] = True
                self.market_allowed[city] = True
                self.Capital += capital_info[city]['market_cap']
                self.updateTotalVP(capital_info[city]['market_cap'])
                output(f"You successfully purchased a market in {city}!", 'green')
                output(f"+{capital_info[city]['market_cap']} Capital.", 'green')
                output(f"+{capital_info[city]['return']} coins at the end of every round you tend the market. Must be in {city} at round end.", 'green')
            self.coins -= cost
            self.checkGameEnd()
            exitActionLoop()()
        if barter is None:
            output(f"This will cost {cost}. Attempt to Barter (+1 Fatigue)?", 'blue')
            actions = {'Yes':partial(self.purchase, asset, city, True), 'No':partial(self.purchase, asset, city, False)}
            actionGrid(actions, False)
        elif barter and (not self.already_asset_bartered):
            if self.coins < cost:
                output("You can't effectively barter when your funds are low!", 'yellow')
                exitActionLoop(amt=0)()
                return
            self.already_asset_bartered = True
            bartering = self.activateSkill("Bartering")
            # Bartering home is more effective than bartering for market
            r = min([cost-1, rbtwn(0, bartering if asset == 'home' else bartering//2)]) # Has to cost at least a coin!
            if r > 0:
                output(f"Your bartering will save you {r} coins. Complete purchase?", 'blue')
            else:
                output("Your bartering failed. Still complete the purchase?", 'blue')
            actions = {'Yes':partial(make_purchase, asset, city, cost-r), 'No':exitActionLoop(amt=0)}
            self.fatigue += 1 # Take a fatigue for bartering
            actionGrid(actions, False)
        elif barter and self.already_asset_bartered:
            output("You can't barter again this action!", 'yellow')
            return
        else:
            make_purchase(asset, city, cost)
    def get_worker(self, city, _=None):
        if self.paused:
            return
        elif not self.markets[city]:
            output("You need to purchase a market first!","yellow")
            return
        elif self.workers[city]:
            output("You already manage a worker here!","yellow")
            return
        excavating = self.activateSkill("Excavating")
        r = rbtwn(0, 6, None, excavating, 'Excavating ')
        if r <= excavating:
            self.useSkill("Excavating")
            output("You found a potential worker")
            persuasion = self.activateSkill("Persuasion")
            r = rbtwn(0, 6, None, persuasion, 'Persuasion ')
            if r <= persuasion:
                self.useSkill("Persuasion")
                output("You convinced them to work for you!", 'green')
                self.workers[city] = True
            else:
                output("You were unable to convince the worker", 'red')
        else:
            output("You were unable to find a worker", 'red')
        exitActionLoop()()
    def getIncome(self,_=None):
        notwaring = notAtWar(Skirmishes[0])
        def receive_market_from_tending(city):
            # Check if bartering will give you extra coins - does not count as an activation!
            r = rbtwn(0, self.skills["Bartering"])
            self.coins += capital_info[city]['return'] + r # Get the income plus the bartering effort
            self.updateTitleValue('entrepreneur', capital_info[city]['return'] + r)
            brtr_msg = '' if r <= 0 else f" plus {r} coins from bartering!"
            self.tended_market[city] = False
            coins_plurality = 'coin' if capital_info[city]['return'] == 1 else 'coins'
            output(f"You received {capital_info[city]['return']} {coins_plurality} from your market{brtr_msg}!", 'green')
        for city in cities:
            if city in notwaring:
                # If you have the city
                if self.markets[city]:
                    # If you tended the market (for 1 action) this round and still in the city.
                    if (self.currenttile.tile == city) and self.tended_market[city]:
                        receive_market_from_tending(city)
                    # Otherwise check if you have automated the city
                    elif self.workers[city]:
                        if capital_info[city]['return'] <= 1:
                            self.bank[city] += 0.5 # only half a coin goes to the bank.
                        else:
                            self.bank[city] += capital_info[city]['return'] - 1 # 1 coin goes to the worker and money sent to bank
            elif self.tended_market[city]:
                receive_market_from_tending(city)
        self.update_mainStatPage()
    def invest(self, item=None, replace=False, _=None):
        if self.paused:
            return
        if self.currenttile.tile not in village_invest:
            return
        city = self.currenttile.neighbortiles.intersection(set(cities)).pop()
        cost = capital_info[city]['invest']
        if self.coins < cost:
            output("Insufficient funds to invest!", 'yellow')
            return
        rounds = 12 - capital_info[city]['invest']
        rounds = round(rounds + (6 - rounds)/1.9) # Minimum of 5 rounds per good, max 7 rounds per good.
        if self.currenttile.tile == 'village3':
            if 'village3' in self.villages[city]:
                output("You have already invested in the village", 'yellow')
                return
            item = city+' cloth' if (city+' cloth') in clothSpecials else 'luxurious cloth'
            self.villages[city]['village3'] = [item, rounds, 1] # Item, Rounds until item is received, number of items to receive
            self.coins -= cost
            self.Capital += 1
            self.updateTotalVP(1)
            exitActionLoop(amt=1)()
            return
        categ, price = getItemInfo(item)
        if village_invest[self.currenttile.tile] != categ:
            output(f"You can't invest that item here! They take {village_invest[self.currenttile.tile]} items.", 'yellow')
            return
        if self.currenttile.tile in self.villages[city]:
            if not replace:
                output("Replace old investment with new one?", 'blue')
                actions = {'Yes': partial(self.invest, item, True), 'No': exitActionLoop(amt=0)}
                actionGrid(actions, False)
                return
            else:
                output("Replacing old investment with new one!")
                if self.villages[city][self.currenttile.tile][2] > 1:
                    self.addItem(item, 1)
        self.addItem(item, -1)
        self.villages[city][self.currenttile.tile] = [item, rounds, 2] # A newly produced item and your original (borrowed) item
        self.Capital += 1
        self.updateTotalVP(1)
        self.coins -= cost
        exitActionLoop(amt=1)()
    def add_coins(self, amt):
        self.coins += amt
        self.update_frontStats()
    def storeInvestment(self, city, village, item, amt):
        for i in range(amt):
            if village in self.awaiting[city]:
                self.awaiting[city][village].append(item)
            else:
                self.awaiting[city][village] = [item]
    def getAwaiting(self):
        if self.currenttile.tile in {'village1','village2','village3','village4','village5'}:
            city = self.currenttile.neighbortiles.intersection(set(cities)).pop()
            if self.currenttile.tile in self.awaiting[city]:
                for item in self.awaiting[city][self.currenttile.tile]:
                    self.addItem(item, 1, [city, self.currenttile.tile])
    def receiveInvestments(self):
        for city in self.villages:
            for v in self.villages[city]:
                self.villages[city][v][1] -= 1
                if self.villages[city][v][1] == 0:
                    if (self.currenttile.tile == v) and (city in self.currenttile.neighbortiles):
                        # If the player is on the exact village they invested with then add the items directly to their inventory
                        self.addItem(self.villages[city][v][0], self.villages[city][v][2], store=[city, v])
                    else:
                        # Otherwise store the investment
                        self.storeInvestment(city, v, self.villages[city][v][0], self.villages[city][v][2])
                    rounds = 12 - capital_info[city]['invest']
                    self.villages[city][v][1] = max([1, round(rounds + (6 - rounds)/1.9) - capital_info[city]['efficiency']])
                    self.villages[city][v][2] = 1 # Have one waiting after the round count is over
    def addItem(self, item, amt, store=False, skip_update=False):
        if amt > 0:
            if self.item_count >= self.max_capacity:
                if store:
                    self.storeInvestment(store[0], store[1], item, amt)
                    output("Not enough inventory space! Sent to awaiting!", 'yellow')
                else:
                    output("Cannot add anymore items!",'yellow')
                self.update_mainStatPage()
                return
            elif (self.item_count + amt) > self.max_capacity:
                origamt = amt
                amt = self.max_capacity - self.item_count
                if store:
                    self.storeInvestment(store[0], store[1], item, origamt-amt)
                    output("Some items had to be stored for later!")
                else:
                    output("Could not add all the items!",'yellow')
        else:
            if item not in self.items:
                output(f"{item} doesn't exist in the inventory!", 'yellow')
                return
            elif (item == 'fruit') and ((self.items[item] - amt) <= 0) and (self.PlayerTrack.Quest.quests[1, 8].status == 'started') and hasattr(self.PlayerTrack.Quest.quests[1, 8], 'has_mammoth') and self.PlayerTrack.Quest.quests[1, 8].has_mammoth:
                output("You lost your fruit! The baby mammoth stops following you! You will need to find it again.", 'red')
                self.PlayerTrack.Quest.quests[1, 8].has_mammoth = False
        self.item_count += amt
        if item in self.items:
            self.items[item] += amt
        else:
            self.items[item] = amt
        if self.items[item]<=0:
            self.items.pop(item)
        if not skip_update:
            self.update_mainStatPage()

#%% Player Track: Armory + Crafting
class ArmoryTable(GridLayout):
    def __init__(self, relative_height=0.2, fsize=12, **kwargs):
        super().__init__(**kwargs)
        self.P = lclPlayer()
        self.aspace = [[], [], []]
        self.space_items = [0, 0, 0]
        self.cols=1
        self.height = Window.size[1]*relative_height
        self.relative_height = relative_height
        self.size_hint_y = None
        self.fsize = fsize
        self.primary = 1
        self.secondary = 2
        for space in range(3):
            if space == 0:
                B = Button(text=f"{self.P.combatstyle}\nWeapon", font_size=fsize-2, disabled=True, bold=True, background_color=(0.6, 0.6, 0.6, 1), color=(0.2, 0.2, 0.2, 1))
            else:
                txt, dsbld, clr = ("Equipped",True,(0,0,0,1)) if space==1 else ("Equip",False,(1,1,1,1))
                B = Button(text=txt, disabled=dsbld, font_size=fsize-1, color=clr)#, background_color=(1.3,1.3,1.3,1))
                B.bind(on_press=partial(self.change_primary))
            self.aspace[space].append(B)
            R = 2 if space == 0 else 4
            for i in range(R):
                txt, clr, bkg = ('Rank1 (50% Success)', (1,1,0,1), (1,1,1,1)) if i == 0 else ('Locked', (1,0,0,1), (0.2, 0.2, 0.2, 1))
                B = Button(text=txt, disabled=True, font_size=fsize, background_color=bkg, color=clr)
                B.bind(on_press=partial(self.rmv_slot, space, i+1, False))
                self.aspace[space].append(B)
            B = Button(text='', disabled=True, background_color=(1,1,1,0), markup=True)
            B.bind(on_press=partial(self.sell, space, None))
            self.aspace[space].append(B)
        for space in range(len(self.aspace)):
            rowGrid = GridLayout(cols = len(self.aspace[space]))
            for slot in range(len(self.aspace[space])):
                rowGrid.add_widget(self.aspace[space][slot])
            self.add_widget(rowGrid)
    def change_primary(self, _=None):
        if self.P.currenttile.tile in cities:
            self.primary = 1 if self.primary==2 else 2
            self.secondary = 1 if self.secondary==2 else 2
            self.aspace[self.secondary][0].disabled=False
            self.aspace[self.secondary][0].text = ""
            self.aspace[self.secondary][0].color = (1,1,1,1)
            self.aspace[self.primary][0].disabled=True
            self.aspace[self.primary][0].text="Equipped"
            self.aspace[self.primary][0].color = (0, 0, 0, 1)
            for B in self.aspace[self.secondary][1:-1]:
                if B.text in ore_properties:
                    atr, amt = ore_properties[B.text]
                    self.P.applyBoost(f'Def-{atr}', -amt, False)
            for B in self.aspace[self.primary][1:-1]:
                if B.text in ore_properties:
                    atr, amt = ore_properties[B.text]
                    self.P.applyBoost(f'Def-{atr}', amt, False)
            self.P.update_mainStatPage()
        else:
            output("You can only change equipped armor in a city!", 'yellow')
    def enable_selling(self):
        for space in range(len(self.aspace)):
            if self.aspace[space][-1].text == '':
                continue
            self.aspace[space][-1].disabled = False
            self.aspace[space][-1].background_color = (1,1,1,1)
    def disable_selling(self):
        for space in range(len(self.aspace)):
            self.aspace[space][-1].disabled = True
            self.aspace[space][-1].background_color = (1,1,1,0)
    def sell_value(self, space):
        ttl, sm = set(), 0
        for slot in range(1, 3 if space==0 else 5):
            if self.aspace[space][slot].text in gameItems["Smithing"]:
                categ, price = getItemInfo(self.aspace[space][slot].text)
                ttl.add(self.aspace[space][slot].text) # Total unique
                sm += sellPrice[price]
        # Smithing price calculator
        if space == 0:
            return int(len(ttl)**3//1.1) + sm
        else:
            return int(len(ttl)**2//1.5) + sm
    def assign_sell_value(self, space):
        sellprice = self.sell_value(space)
        if self.space_items[space] == 0:
            self.aspace[space][-1].disabled = True
            self.aspace[space][-1].text = ''
            self.aspace[space][-1].background_color = (1,1,1,0)
        else:
            clr = 'ffff75' if self.P.bartering_mode == 0 else '00ff75'
            self.aspace[space][-1].text = f'Sell: [color={clr}]{sellprice+self.P.bartering_mode}[/color]'
            if (self.P.currenttile.tile in cities) or (self.P.currenttile.trader_rounds > 0):
                self.enable_selling()
    def reassign_lbl(self, space, slot, update_stat=False):
        smithing = self.P.skills["Smithing"]
        def adjustment(txt, clr=(1,1,1,1), bkg=(1,1,1,1)):
            if self.aspace[space][slot].text in ore_properties:
                atr, amt = ore_properties[self.aspace[space][slot].text]
                if space == 0:
                    # Remove attack boost
                    if atr == self.P.combatstyle:
                        self.P.applyBoost('Attack', -amt, update_stat)
                    else:
                        # Somehow an invalid ore got into player's weapon?
                        output(f"Invalid ore: {self.aspace[space][slot].text} found in weapon slot", 'red')
                else:
                    self.P.applyBoost(f'Def-{atr}', -amt, update_stat)
            self.aspace[space][slot].text = txt
            self.aspace[space][slot].color = clr
            self.aspace[space][slot].background_color = bkg
        if (slot == 2) and (smithing == 0):
            adjustment('Rank1 (50% Success)', (1, 1, 0, 1))
        elif (space == 0) and (slot == 2):
            # First weaponry slot
            lvready = (smithing - 1)//3 + 1
            adjustment(f'Rank{lvready} Ready')
        elif (space == 0) and (slot == 3):
            # Second weapon slot
            if smithing >= 12:
                adjustment('Rank4 Ready')
            else:
                adjustment('Locked', (1, 0, 0, 1), (0.2, 0.2, 0.2, 1))
        else:
            # Armory slot
            lvready = (smithing - 2)//3 + 1
            slotunlocked = smithing//3 + 1
            if slotunlocked >= slot:
                adjustment(f'Rank{lvready} Ready')
            else:
                adjustment('Locked', (1, 0, 0, 1), (0.2, 0.2, 0.2, 1))
        self.aspace[space][slot].disabled = True
    def update_lbls(self):
        # This function is not called by anything as of yet
        for space in range(len(self.aspace)):
            for slot in range(1, 3 if space==0 else 5):
                if self.aspace[space][slot].text in gameItems['Smithing']:
                    self.reassign_lbl(space, slot)
        self.P.update_mainStatPage()
    def rmv(self, space, take_minor_action=False):
        for slot in range(1, 3 if space==0 else 5):
            if self.aspace[space][slot].text in gameItems['Smithing']:
                self.P.item_count -= 1
            self.reassign_lbl(space, slot)
        self.P.update_mainStatPage()
        self.space_items[space] = 0
        self.assign_sell_value(space)
        if take_minor_action:
            exitActionLoop('minor', 1)()
    def rmv_slot(self, space, slot, confirmed=False, _=None):
        if self.P.paused:
            return
        if self.aspace[space][slot].text not in gameItems['Smithing']:
            output("Space is already empty!", 'yellow')
        elif not confirmed:
            output("Would you like to remove and destroy this item?", "blue")
            actionGrid({"Yes":partial(self.rmv_slot, space, slot, True), "No":exitActionLoop(amt=0)}, False, False)
        else:
            prev = None
            for i in np.arange(slot, 3 if space==0 else 5)[::-1]:
                if self.aspace[space][slot].text not in gameItems['Smithing']:
                    continue
                if prev is None:
                    prev = self.aspace[space][slot].text
                    self.reassign_lbl(space, slot)
                else:
                    newprev = self.aspace[space][slot].text
                    self.aspace[space][slot].text = prev
                    prev = newprev
            output(f"{prev} is destroyed.", 'yellow')
            self.space_items[space] -= 1
            self.P.item_count -= 1
            self.assign_sell_value(space)
            self.P.update_mainStatPage()
    def saveTable(self):
        for space in range(3):
            for slot in range(1, 3 if space==0 else 5):
                self.P.aspace[space][slot-1] = self.aspace[space][slot].text if self.aspace[space][slot].text in gameItems['Smithing'] else None
    def loadTable(self):
        for space in range(3):
            for slot in range(1, 3 if space==0 else 5):
                if self.P.aspace[space][slot-1] is not None:
                    self.confirmed_add_slot(self.P.aspace[space][slot-1], space, slot, False)
            self.assign_sell_value(space)
    def confirmed_add_slot(self, item, space, slot, from_items):
        if (not from_items) and (self.P.item_count >= self.P.max_capacity):
            output("Inventory is full!", 'yellow')
        else:
            self.aspace[space][slot].text = item
            self.aspace[space][slot].color = (0, 0.6, 0, 1)
            self.aspace[space][slot].disabled = True
            self.space_items[space] += 1
            if from_items: self.P.addItem(item, -1)
            self.P.item_count += 1
    def add_slot(self, item, space=None, cost=None, barter=None, _=None):
        if self.P.paused:
            return
        if self.P.currenttile.tile not in cities:
            output("You can only smith in a city!", 'yellow')
            return
        smitherslvl = city_info[self.P.currenttile.tile]['Smithing']
        categ, price = getItemInfo(item)
        reqlvl = price2smithlvl[price]
        # 2 is the base rental cost -- can be reduced by quest
        rentalcost = 0 if self.P.free_smithing_rent and (self.P.currenttile.tile == self.P.birthcity) else max([2 - int(self.P.bartering_mode>0), 0]) # Can't barter more than 1 coin (because smith is not a trader)
        fullcost = (price*reqlvl) + 4 - int(self.P.bartering_mode>0)
        if categ != 'Smithing':
            output("You cannot smelt that item.", 'yellow')
            return
        def attempt_smith(success, slot, smithing_on_own):
            if success:
                atr, amt = ore_properties[item]
                if space == 0:
                    # Assign Attack Boost
                    if atr != self.P.combatstyle:
                        output(f"This ore is for {atr} while your attack style is {self.P.combatstyle}!", 'yellow')
                        exitActionLoop(amt=0)()
                        return
                    else:
                        self.P.applyBoost('Attack', amt, False)
                        output(f"Smithing successful! Attack boosted by {amt}", 'green')
                else:
                    # Assign Defense Boost
                    self.P.applyBoost(f'Def-{atr}', amt, False)
                    output(f"Smithing successful! Def-{atr} boosted by {amt}", 'green')
                self.confirmed_add_slot(item, space, slot, True)
                if smithing_on_own:
                    exitActionLoop()()
                else:
                    # Minor action if hiring smither
                    exitActionLoop('minor')()
            elif smithing_on_own:
                self.P.addItem(item, -1)
                output("Failed. Item is Destroyed.", 'red')
                exitActionLoop()()
            else:
                output("Smither's level is not high enough for this action", 'yellow')
                exitActionLoop(amt=0)()
        if space is None:
            output("Which smithing piece would you like to add it to?", 'blue')
            actions = {'Equipped Armor':partial(self.add_slot, item, self.primary, None, None), 'Non-Equipped Armor':partial(self.add_slot, item, self.secondary, None, None), 'Cancel':exitActionLoop(amt=0)}
            # Only allow adding to weapon if combatstyle matches
            if ore_properties[item][0] == self.P.combatstyle: actions['Weapon'] = partial(self.add_slot, item, 0, None, None)
            actionGrid(actions, False)
        elif (cost is None) and (not self.P.activated_bartering) and (barter is None):
            output(f"Rental cost is {rentalcost} or smith charges {fullcost}. Will you attempt to barter?", 'blue')
            actionGrid({'Yes':partial(self.add_slot, item, space, None, True),
                        'No':partial(self.add_slot, item, space, None, False),
                        'Cancel':exitActionLoop(amt=0)},False)
        elif cost is None:
            if barter and (not self.P.activated_bartering):
                bartering = self.P.activateSkill("Bartering")
                r = rbtwn(1, 12, None, bartering, 'Bartering ')
                if r <= bartering:
                    self.P.useSkill("Bartering")
                    output("Bartered with smith and reduced their cost!")
                    self.P.bartering_mode = 1
                    rentalcost = max([rentalcost - 1, 0])
                    fullcost -= 1
                else:
                    output("Unable to successfully barter with smith.",'yellow')
            clr = 'ffff75' if self.P.bartering_mode == 0 else '00ff75'
            actionGrid({f'Rent Facility: [color={clr}]{rentalcost}[/color]':partial(self.add_slot, item, space, rentalcost),
                        f'Hire Smither Lv{smitherslvl}: [color={clr}]{fullcost}[/color]':partial(self.add_slot, item, space, fullcost),
                        'Cancel':exitActionLoop(amt=0)}, False, False)
        else:
            if self.P.coins < cost:
                output("Insufficient funds!", 'yellow')
                exitActionLoop(amt=0)()
                return
            self.P.coins -= cost
            if cost <= 2:
                # Meaning they have claimed to do it themselves
                smithing = self.P.activateSkill("Smithing")
                smithing_on_own = True
            else:
                smithing = smitherslvl
                smithing_on_own = False
            if space==0:
                # Weapon slot
                if self.space_items[space] == 0:
                    lvready = max([1, (smithing - 1)//3 + 1])
                    if smithing == 0:
                        attempt_smith(rbtwn(0,1) * (lvready >= reqlvl), 1, smithing_on_own)
                    else:
                        attempt_smith(lvready>=reqlvl, 1, smithing_on_own)
                else:
                    attempt_smith(smithing>=12, 2, smithing_on_own)
            else:
                # Armor slot
                lvready = max([1, (smithing - 2)//3 + 1])
                slotunlocked = smithing//3 + 1
                if (self.space_items[space] == 0) and smithing in {0,1}:
                    attempt_smith(rbtwn(0,1) * (lvready >= reqlvl), 1, smithing_on_own)
                elif ((self.space_items[space]+1) <= slotunlocked):
                    attempt_smith(lvready >= reqlvl, self.space_items[space]+1, smithing_on_own)
                else:
                    attempt_smith(False, self.space_items[space]+1, smithing_on_own)
            self.assign_sell_value(space)
    def sell(self, space, barter=None, _=None):
        if self.P.paused:
            return
        elif (self.P.currenttile.tile not in cities) and (self.P.currenttile.trader_rounds==0):
            # If you are not in a city or not in any trading places, then you can't sell the item
            return
        elif (self.space_items[space]==1) and (self.aspace[space][1].text in self.P.unsellable):
            output("You can't sell that item this turn!", 'yellow')
            return
        sellprice = self.sell_value(space)
        if (barter is None) and (not self.P.activated_bartering):
            output("Would you like to barter?",'blue')
            actionGrid({'Yes':partial(self.sell_craft, space, True), 'No':partial(self.sell_craft, space, False), 'Cancel':exitActionLoop(amt=0)}, False, False)
        elif barter == True:
            self.P.activated_bartering = True
            bartering = self.P.activateSkill("Bartering")
            r = rbtwn(1, 12, None, bartering, 'Bartering ')
            if r <= bartering:
                self.P.useSkill("Bartering")
                output("You successfully Barter", 'green')
                if (bartering > 8) and (self.player.trader_rounds>0):
                    self.P.bartering_mode = 2
                    sellprice += 2
                else:
                    self.P.bartering_mode = 1
                    sellprice += 1
                self.P.coins += sellprice
                output(f"Sold smithed piece for {sellprice}.")
                self.P.updateTitleValue('merchant', sellprice)
                self.rmv(space, True)
            else:
                output("You failed to barter, sell anyway?", 'yellow')
                actionGrid({'Yes':partial(self.sell, space, False), 'No':exitActionLoop('minor', 0, False)}, False, False)
        else:
            sellprice += self.P.bartering_mode
            output(f"Sold smithed piece for {sellprice}.")
            self.P.coins += sellprice
            self.P.updateTitleValue('merchant', sellprice)
            self.rmv(space, True)

class CraftingTable(GridLayout):
    def __init__(self, relative_height = 0.15, fsize=12, **kwargs):
        super().__init__(**kwargs)
        self.P = lclPlayer()
        self.cols = 9
        self.height = Window.size[1]*relative_height
        self.relative_height = relative_height
        self.size_hint_y = None
        self.fsize=fsize
        self.cspace = [[], []]
        self.space_items = [0, 0]
        for space in range(2):
            B = Button(text=f"Craft {space+1}", disabled=True, font_size=fsize-2, bold=True, underline=True, background_color=(1,1,1,0), color=(0.4,0.4,0.4,1))
            self.add_widget(B)
            self.cspace[space].append(B)
            for i in range(7):
                if i == 0:
                    txt, clr = 'Unlocked', (1,1,1,1)
                elif i == 1:
                    txt, clr = '20% Success', (1,1,0,1)
                else:
                    txt, clr = 'Locked', (1,0,0,1)
                bkg = (0.2, 0.2, 0.2, 1) if txt == 'Locked' else (1, 1, 1, 1)
                B = Button(text=txt, disabled=True, font_size=fsize, background_color=bkg, color=clr)
                B.bind(on_press=partial(self.rmv_slot, space, i+1, False))
                self.cspace[space].append(B)
                self.add_widget(B)
            B = Button(text='', disabled=True, background_color=(1,1,1,0), markup=True)
            B.bind(on_press=partial(self.sell_craft, space, None))
            self.cspace[space].append(B)
            self.add_widget(B)
    def enable_selling(self):
        for space in range(len(self.cspace)):
            if self.cspace[space][-1].text == '':
                continue
            self.cspace[space][-1].disabled = False
            self.cspace[space][-1].background_color = (1,1,1,1)
    def disable_selling(self):
        for space in range(len(self.cspace)):
            self.cspace[space][-1].disabled = True
            self.cspace[space][-1].background_color = (1,1,1,0)
    def sell_value(self, space):
        ttl, sm = set(), 0
        for slot in range(1, 8):
            if self.cspace[space][slot].text not in {'Unlocked', '20% Success', '50% Success', 'Locked'}:
                categ, price = getItemInfo(self.cspace[space][slot].text)
                ttl.add(self.cspace[space][slot].text) # Total Unique
                sm += sellPrice[price]
        # Crafting price calculator
        return int(len(ttl)**2//2) + sm
    def assign_sell_value(self, space):
        sellprice = self.sell_value(space)
        if self.space_items[space] == 0:
            self.cspace[space][-1].disabled = True
            self.cspace[space][-1].text = ''
            self.cspace[space][-1].background_color = (1,1,1,0)
        else:
            clr = 'ffff75' if self.P.bartering_mode == 0 else '00ff75'
            self.cspace[space][-1].text = f'Sell: [color={clr}]{sellprice+self.P.bartering_mode}[/color]'
            if (self.P.currenttile.tile in cities) or (self.P.currenttile.trader_rounds > 0):
                self.enable_selling()
    def reassign_lbl(self, space, slot):
        crafting = self.P.skills["Crafting"]
        unlocked = crafting/2 + 1
        if (slot == 2) and (crafting == 0):
            self.cspace[space][slot].text = '20% Success'
            self.cspace[space][slot].color = (1, 1, 0, 1)
            self.cspace[space][slot].background_color = (1, 1, 1, 1)
        elif (unlocked - slot) >= 0:
            if int(unlocked) == unlocked:
                # Gauranteed Unlocked
                self.cspace[space][slot].color = (1, 1, 1, 1)
                self.cspace[space][slot].text = 'Unlocked'
            else:
                # 50% success unlocked
                self.cspace[space][slot].text = '50% Success'
                self.cspace[space][slot].color = (1, 1, 0, 1)
            self.cspace[space][slot].background_color = (1, 1, 1, 1)
        else:
            self.cspace[space][slot].text = 'Locked'
            self.cspace[space][slot].color = (1, 0, 0, 1)
        self.cspace[space][slot].disabled = True
    def update_lbls(self):
        for space in range(len(self.cspace)):
            for slot in range(1, 8):
                if self.cspace[space][slot].text in {'20% Success', '50% Success', 'Locked'}:
                    self.reassign_lbl(space, slot)
    def rmv_craft(self, space, take_minor_action=False):
        for slot in range(1, 8):
            if self.cspace[space][slot].text not in {'Unlocked', '50% Success', '20% Success', 'Locked'}:
                self.P.item_count -= 1
            self.reassign_lbl(space, slot)
        self.space_items[space] = 0
        self.assign_sell_value(space)
        if take_minor_action:
            exitActionLoop('minor', 1)()
    def rmv_slot(self, space, slot, confirmed=False, _=None):
        if self.P.paused:
            return
        if self.cspace[space][slot].text in {'Unlocked', '50% Success', '20% Success', 'Locked'}:
            output("Space is already empty!", 'yellow')
        elif self.space_items[space] == 1:
            # The "crafted item" is still just an item, so just put back into item list
            self.P.item_count -= 1
            self.P.addItem(self.cspace[space][slot].text, 1)
            self.reassign_lbl(space, slot)
        elif not confirmed:
            output("Would you like to remove and destroy this item?", "blue")
            actionGrid({"Yes":partial(self.rmv_slot, space, slot, True), "No":exitActionLoop(amt=0)}, False, False)
        else:
            prev = None
            for i in np.arange(slot, 8)[::-1]:
                if self.cspace[space][slot].text in {'Unlocked', '50% Success', '20% Success', 'Locked'}:
                    continue
                if prev is None:
                    prev = self.cspace[space][slot].text
                    self.reassign_lbl(space, slot)
                else:
                    newprev = self.cspace[space][slot].text
                    self.cspace[space][slot].text = prev
                    prev = newprev
            output(f"{prev} is destroyed.", 'yellow')
            self.space_items[space] -= 1
            self.P.item_count -= 1
            self.assign_sell_value(space)
    def saveTable(self):
        for space in range(2):
            for slot in range(1, 8):
                self.P.cspace[space][slot-1] = self.cspace[space][slot].text if self.cspace[space][slot].text in gameItems['Crafting'] else None
    def loadTable(self):
        for space in range(2):
            for slot in range(1, 8):
                if self.P.cspace[space][slot-1] is not None:
                    self.confirmed_add_slot(self.P.cspace[space][slot-1], space, slot, False)
    def confirmed_add_slot(self, item, space, slot, from_items):
        if (not from_items) and (self.P.item_count >= self.P.max_capacity):
            output("Not enough inventory space to add to craft!", 'yellow')
        else:
            self.cspace[space][slot].text = item
            self.cspace[space][slot].color = (0, 0.6, 0, 1)
            self.cspace[space][slot].disabled = False
            self.space_items[space] += 1
            if from_items: self.P.addItem(item, -1)
            self.P.item_count += 1
    def add_craft(self, item, space=None, _=None):
        if self.P.paused:
            return
        if space is None:
            output("Which craft would you like to add it to?", 'blue')
            actionGrid({'Craft 1':partial(self.add_craft, item, 0), 'Craft 2':partial(self.add_craft, item, 1), 'Cancel':exitActionLoop(amt=0)}, False, False)
        else:
            if self.space_items[space] == 0:
                self.cspace[space][1].text = item
                self.cspace[space][1].color = (0, 0.6, 0, 1)
                self.cspace[space][1].disabled = False
                self.space_items[space] += 1
                self.P.addItem(item, -1)
                self.P.item_count += 1
                exitActionLoop(amt=0)()
            else:
                crafting = self.P.activateSkill("Crafting")
                def attempt_craft(success, slot):
                    if success:
                        self.P.useSkill("Crafting")
                        self.confirmed_add_slot(item, space, slot, True)
                    else:
                        output("Failed. Item is Destroyed.", 'red')
                        self.P.addItem(item, -1)
                    exitActionLoop(amt=0)()
                for slot in range(1, 8):
                    if self.cspace[space][slot].text not in {'Unlocked', '20% Success', '50% Success'}:
                        continue
                    requiredlvl = (slot - 1)*2-1.5 # Converting level to slot.
                    if (crafting==0) and (slot==2):
                        attempt_craft(rbtwn(1,5)==1, slot)
                    elif crafting > requiredlvl:
                        if (crafting - requiredlvl) == 0.5:
                            attempt_craft(rbtwn(0,1), slot)
                        else:
                            attempt_craft(True, slot)
                    else:
                        attempt_craft(False, slot)
                    break
            self.assign_sell_value(space)
    def sell_craft(self, space, barter=None, _=None):
        if self.P.paused:
            return
        elif (self.P.currenttile.tile not in cities) and (self.P.currenttile.trader_rounds==0):
            # If you are not in a city or not in any trading places, then you can't sell the item
            return
        elif (self.space_items[space]==1) and (self.cspace[space][1].text in self.P.unsellable):
            output("You can't sell that item this turn!", 'yellow')
            return
        sellprice = self.sell_value(space)
        if (barter is None) and (not self.P.activated_bartering):
            output("Would you like to barter?",'blue')
            actionGrid({'Yes':partial(self.sell_craft, space, True), 'No':partial(self.sell_craft, space, False), 'Cancel':exitActionLoop(amt=0)}, False, False)
        elif barter == True:
            self.P.activated_bartering = True
            bartering = self.P.activateSkill("Bartering")
            r = rbtwn(1, 12, None, bartering, 'Bartering ')
            if r <= bartering:
                self.P.useSkill("Bartering")
                output("You successfully Barter", 'green')
                if (bartering > 8) and (self.player.trader_rounds>0):
                    self.P.bartering_mode = 2
                    sellprice += 2
                else:
                    self.P.bartering_mode = 1
                    sellprice += 1
                self.P.coins += sellprice
                output(f"Sold craft {space+1} for {sellprice}.")
                self.P.updateTitleValue('merchant', sellprice)
                self.rmv_craft(space, True)
            else:
                output("You failed to barter, sell anyway?", 'yellow')
                actionGrid({'Yes':partial(self.sell_craft, space, False), 'No':exitActionLoop('minor', 0, False)}, False, False)
        else:
            sellprice += self.P.bartering_mode
            output(f"Sold craft {space+1} for {sellprice}.")
            self.P.coins += sellprice
            self.P.updateTitleValue('merchant', sellprice)
            self.rmv_craft(space, True)
    def getItems(self, space):
        items = set()
        for B in self.cspace[space][1:-1]:
            if B.text in gameItems['Crafting']:
                items.add(B.text)
        return items

#%% Player Track: Quests

def getQuest(stage, mission):
    P = lclPlayer()
    return P.PlayerTrack.Quest.quests[stage, mission]

def FindPet(_=None):
    P = lclPlayer()
    if P.paused:
        return
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 2, None, excavating, 'Excavating ')
    if r <= excavating:
        P.useSkill("Excavating")
        output("You found the pet! You get 1 coin!", 'green')
        P.coins += 1
        P.PlayerTrack.Quest.update_quest_status((1, 1), 'complete')

    else:
        output("Unable to find the pet.", 'yellow')
    exitActionLoop()()


def CleanHome(_=None):
    P = lclPlayer()
    if P.paused:
        return
    B = getQuest(1, 2)
    B.count = B.count + 1 if hasattr(B, 'count') else 1
    output(f"You spent an action cleaning the home (Total: {B.count})")
    if B.count >= 4:
        P.addItem('string', 1)
        P.addItem('beads', 1)
        P.PlayerTrack.Quest.update_quest_status((1, 2), 'complete')
    exitActionLoop()()


def GaurdHome(_=None):
    P = lclPlayer()
    if P.paused:
        return
    B = getQuest(1, 3)

    def Reward():
        amt = P.get_bonus(1)
        actions = {'Tin': getItem('tin', amt, action_amt=0),
                   'Iron': getItem('iron', amt, action_amt=0),
                   'Lead': getItem('lead', amt, action_amt=0),
                   'Copper': getItem('copper', amt, action_amt=0)}
        P.PlayerTrack.Quest.update_quest_status((1, 3), 'complete')
        output("The owner wants to reward you for gaurding the house! Choose one of the following:", 'blue')
        actionGrid(actions, False)

    def Consequence():
        P.PlayerTrack.Quest.update_quest_status((1, 3), 'failed')

    B.count = B.count + 1 if hasattr(B, 'count') else 1
    output("You spent an action gaurding the house")
    if B.count >= 2:
        encounter('Robber', [6, 6], [P.combatstyle], Reward, consequence=Consequence,
                  background_img='images\\resized\\background\\cottage.png')
    else:
        exitActionLoop()()


def OfferCraft(_=None):
    P = lclPlayer()
    if P.paused:
        return

    def Offer(space, _=None):
        sellprice = P.PlayerTrack.craftingTable.sell_value(space)
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 4, None, sellprice + persuasion, 'Persuasion ')
        if r <= (sellprice + persuasion):
            def lvlUp(skill, _=None):
                P.updateSkill(skill, 1, 6)
                exitActionLoop()()

            P.activateSkill("Persuasion")
            P.PlayerTrack.craftingTable.rmv_craft(space)
            output("The boy accepted your craft!", 'green')
            P.PlayerTrack.Quest.update_quest_status((1, 4), 'complete')
            if (P.skills['Crafting'] < 6) or (P.skills['Persuasion'] < 6):
                output("Choose a skill to level up:", 'blue')
                actions = {skill: partial(lvlUp, skill) for skill in ['Crafting', 'Persuasion']}
                actionGrid(actions, False)
            else:
                output("Your Crafting and Persuasion are already level 6+", 'yellow')
                exitActionLoop()()
        else:
            output("Unable to convince the boy to accept your craft.", 'yellow')
            exitActionLoop()()

    actions = {'Cancel': exitActionLoop(amt=0)}
    if P.PlayerTrack.craftingTable.space_items[0] > 1:
        actions["Craft 1"] = partial(Offer, 0)
    if P.PlayerTrack.craftingTable.space_items[1] > 1:
        actions["Craft 2"] = partial(Offer, 1)
    actionGrid(actions, False)


def SpareWithBoy(_=None):
    P = lclPlayer()
    if P.paused:
        return
    attack = P.combat[P.attributes["Attack"]]
    r = rbtwn(1, 5, None, attack, 'Sparring ')
    if r <= attack:
        output("The young warrior is satisfied with your playful duel! Impact stability increases by 1.", 'green')
        P.stability_impact += 1
        P.PlayerTrack.Quest.update_quest_status((1, 5), 'complete')
    exitActionLoop()()


def checkBook():
    P = lclPlayer()
    for item in P.items:
        if 'book' in item:
            return True
    return False


def OfferBook(_=None):
    P = lclPlayer()
    if P.paused:
        return

    def Offer(book, _=None):
        P.addItem(book, -1)
        P.PlayerTrack.Quest.update_quest_status((1, 6), 'complete')
        exitActionLoop()()

    actions = {"Cancel": exitActionLoop(amt=0)}
    for item in P.items:
        if 'book' in item:
            actions[item] = partial(Offer, item)
    actionGrid(actions, False)


def OfferSand(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.addItem('sand', -1)
    P.PlayerTrack.Quest.update_quest_status((1, 7), 'complete')
    exitActionLoop()()


def ZooKeeper(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((1, 8), 'complete')
    exitActionLoop()()


def ApplyCubes(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[2, 1], 'count'):
        P.PlayerTrack.Quest.quests[2, 1].count = 1
        output("Partially applied cubes")
    else:
        P.PlayerTrack.Quest.quests[2, 1].count += 1
        P.addItem('cooling cubes', -1)
        output("You applied the cubes successfully! The smith will now let you rent his facility for free!", 'green')
        P.free_smithing_rent = True
        P.PlayerTrack.Quest.update_quest_status((2, 1), 'complete')
    exitActionLoop()()


def WaitforRobber(_=None):
    P = lclPlayer()
    if P.paused:
        return

    def Reward(_=None):
        output(
            "The Robber is scared off! Market owners will now give you a discount of 1 coin for items costing 5+ coins!",
            'green')
        P.city_discount = 1
        P.city_discount_threshold = 5
        P.PlayerTrack.Quest.update_quest_status((2, 2), 'complete')

    stealth = P.activateSkill("Stealth")
    r = rbtwn(0, 4, None, stealth, "Stealth ")
    if r <= stealth:
        P.useSkill("Stealth")
        output("You find and suprise the Robber!")
        encounter("Robber", [30, 30], [P.combatstyle], Reward, consequence=Reward, encounter=-1,
                  background_img='images\\resized\\background\\city_night.png')
    else:
        output("You were not able to find the Robber", 'yellow')
        exitActionLoop()()


def BeginProtection(_=None):
    P = lclPlayer()
    sorted_cities = sorted(cities)
    i = sorted_cities.index(P.birthcity)
    distances = np.concatenate((connectivity[:i, i], connectivity[i, i:]))
    # If there is a tie, pick a random furthest city
    furthest_city = np.random.choice(np.array(sorted_cities)[distances == np.max(distances)])
    output(f"The Nobleman requests you to protect him until reaching {furthest_city}", 'blue')
    P.PlayerTrack.Quest.quests[2, 3].furthest_city = furthest_city


def MotherSerpent(_=None):
    P = lclPlayer()
    coord, distance, path = P.currenttile.findNearest('pond')
    output("The pond where the Mother Serpent lives is tinted green on the map!", 'blue')
    P.PlayerTrack.Quest.quests[2, 4].pond = coord
    P.parentBoard.gridtiles[coord].color = (1, 1.5, 1, 1)
    P.group["Companion"] = npc_stats(35)


def LibrariansSecret(_=None):
    P = lclPlayer()
    coord, distance, path = P.currenttile.findNearest('oldlibrary')
    output("The old library where they librarian lost his book is tinted green on the map!", 'blue')
    P.PlayerTrack.Quest.quests[2, 5].coord = coord
    P.parentBoard.gridtiles[coord].color = (1, 1.5, 1, 1)
    P.PlayerTrack.Quest.quests[2, 5].has_book = False


def HandOverBook(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((2, 5), 'complete')
    output("You can now gain 8xp from reading books!", 'green')
    P.standard_read_xp = 8


def TheLetter(_=None):
    P = lclPlayer()
    sorted_cities = sorted(cities)
    i = sorted_cities.index(P.birthcity)
    distances = np.concatenate((connectivity[:i, i], connectivity[i, i:]))
    # If there is a tie, pick a random furthest city
    furthest_city = np.random.choice(np.array(sorted_cities)[distances == np.max(distances)])
    output(f"You are requested to take the letter to {furthest_city}", 'blue')
    P.PlayerTrack.Quest.quests[2, 6].furthest_city = furthest_city


def GiveOresToSmith(_=None):
    P = lclPlayer()
    if P.paused:
        return

    def rewards(items, choices_left=2):
        actions = {(item[0].upper() + item[1:]): partial(choose_ore, item, items, choices_left) for item in items}
        actionGrid(actions, False)

    def choose_ore(item, from_items, choices_left, _=None):
        choices_left -= 1
        from_items.pop(item)
        P.addItem(item, 1)
        if choices_left > 0:
            rewards(from_items, choices_left)
        else:
            exitActionLoop()()

    for item in {'aluminum', 'nickel', 'tantalum', 'kevlium'}:
        P.addItem(item, -1)
    P.PlayerTrack.Quest.update_quest_status((2, 7), 'complete')
    output("Choose 2 of the following ores:", 'blue')
    rewards({'titanium', 'chromium', 'tungsten', 'diamond'}, 2)


def PursuadeStealthMaster(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if P.currenttile.tile not in {'enfeir', 'zinzibar'}:
        output("There is no steatlh master to persuade here!", 'yellow')
        exitActionLoop(amt=0)()
    elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'has_book') and P.PlayerTrack.Quest.quests[2, 8].has_book:
        output("You already own the book!", 'yellow')
        exitActionLoop(amt=0)()
    elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds') and (P.PlayerTrack.Quest.quests[2, 8] > 0) and (
            P.currentcoord == P.PlayerTrack.Quest.quests[2, 8].coord):
        output(
            f"You need to wait {P.PlayerTrack.Quest.quests[2, 8].wait_rounds} more rounds for master to write the book",
            'yellow')
        exitActionLoop(amt=0)()
    elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds') and (P.PlayerTrack.Quest.quests[2, 8] == 0) and (
            P.currentcoord == P.PlayerTrack.Quest.quests[2, 8].coord):
        P.PlayerTrack.Quest.quests[2, 8].has_book = True
        output("The stealth master hands you the book, now present it to the quest giver at home!", 'blue')
        exitActionLoop('minor')()
    elif not hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds'):
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 8, None, persuasion, 'Persuasion ')
        if r <= persuasion:
            output(
                "You convinced the master stealth master to write you a special stealth book! Come back in 5 rounds to pick it up!",
                'blue')
            P.PlayerTrack.Quest.quests[2, 8].wait_rounds = 5
            P.PlayerTrack.Quest.quests[2, 8].coord = P.currentcoord
        else:
            output("You fail to convince the master to write you a book", 'yellow')
        exitActionLoop()()
    else:
        # They could only get to this point if they go to the other city and persuade another master
        output("You have already persuaded a stealth master to write you a book!", 'red')
        exitActionLoop(amt=0)()


def PresentStealthBook(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((2, 8), 'complete')
    P.updateSkill('Stealth', 2, 8)
    exitActionLoop('minor')()


def FeedThePoor(_=None):
    P = lclPlayer()
    if P.paused:
        return

    def GiveItem(item, _=None):
        P.addItem(item, -1)
        P.PlayerTrack.Quest.quests[3, 1].food_given += 1
        output(f"You have distributed a total of {P.PlayerTrack.Quest.quests[3, 1].food_given} food!", 'blue')
        if P.PlayerTrack.Quest.quests[3, 1].food_given >= 10:
            P.PlayerTrack.Quest.update_quest_status((3, 1), 'complete')
            output("Your max eating per action increases by 2!", 'green')
            P.max_eating += 2
        exitActionLoop('minor')

    if not hasattr(P.PlayerTrack.Quest.quests[3, 1], 'food_given'):
        P.PlayerTrack.Quest.quests[3, 1].food_given = 0
    actions = {'Cancel': exitActionLoop(amt=0)}
    food_pieces = 0
    for item in P.items:
        if item in {'fruit', 'cooked meat', 'well cooked meat', 'cooked fish', 'well cooked fish'}:
            actions[item] = partial(GiveItem, item)
            food_pieces += 1
    if food_pieces == 0:
        output("You do not have any cooked food or fruit to distribute!", 'yellow')
        exitActionLoop(amt=0)()
    else:
        actionGrid(actions, False)


def resetFitnessTraining(_=None):
    P = lclPlayer()
    output("The three adventurers leave you and you must restart the Fitness Training", 'red')
    for i in range(3):
        P.group.pop(f'Adventurer {i + 1}')
    P.PlayerTrack.Quest.update_quest_status((3, 2), 'not started')


def finishFitnessTraining(_=None):
    P = lclPlayer()
    output("The three adventurers leave you")
    for i in range(3):
        P.group.pop(f'Adventurer {i + 1}')
    P.PlayerTrack.Quest.update_quest_status((3, 2), 'complete')
    for skill in ['Gathering', 'Excavating', 'Survival']:
        P.updateSkill(skill, 1, 8)
    output("Max fatigue increases by 3", 'green')
    P.max_fatigue += 3


def FitnessTraining(_=None):
    P = lclPlayer()
    output(
        "3 Adventurers group up with you. First go to any mountain and find ores worth a total of 5 coins (sell cost) and climb to the top.",
        'blue')
    for i in range(3):
        P.group[f'Adventurer {i + 1}'] = npc_stats(15)
    P.PlayerTrack.Quest.quests[3, 2].total_ore_cost = 0
    P.PlayerTrack.Quest.quests[3, 2].reached_top = False
    P.PlayerTrack.Quest.quests[3, 2].completed_mountain = False
    P.PlayerTrack.Quest.quests[3, 2].meat_collected = 0


def LearnFromMonk(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.quests[3, 3].convinced_monk += 1
    if P.PlayerTrack.Quest.quests[3, 3].convinced_monk >= 3:
        P.PlayerTrack.Quest.update_quest_status((3, 3), 'completed')
        for skill in ['Excavating', 'Survival']:
            P.updateSkill(skill, 1, 12)
    exitActionLoop()()


def SearchForMonster(_=None):
    P = lclPlayer()
    if P.paused:
        return
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 12, None, excavating, 'Excavating ')
    if r <= excavating:
        def monster_flees(rewarded=True, _=None):
            P.PlayerTrack.Quest.update_quest_status((3, 4), 'completed')
            if rewarded:
                P.updateAttribute("Cunning", 1, 8)
                P.updateAttribute("Technique", 1, 8)

        encounter('Monster', [55, 55], ['Physical', 'Elemental', 'Trooper', 'Wizard'], monster_flees,
                  consequence=partial(monster_flees, False),
                  background_img='images\\resized\\background\\city_night.png')
    else:
        output("You fail to find the monster.", 'yellow')
        exitActionLoop()()


def TeachFishing(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[3, 5], 'count'): P.PlayerTrack.Quest.quests[3, 5].count = 0
    gathering = P.activateSkill("Gathering")
    r = rbtwn(1, 8, None, gathering, 'Gathering ')
    if r <= gathering:
        P.useSkill("Gathering")
        P.PlayerTrack.Quest.quests[3, 5].count += 1
        output(f'Success! So far, taught {P.PlayerTrack.Quest.quests[3, 5].count} men to fish', 'blue')
        if P.PlayerTrack.Quest.quests[3, 5].count >= 6:
            P.PlayerTrack.Quest.update_quest_status((3, 5), 'complete')
            P.coins += 8
            P.updateSkill('Gathering', 1, 8)
    else:
        output("Unable to teach a man to fish.", 'yellow')
    exitActionLoop()()


def StartofSkirmish(_=None):
    P = lclPlayer()
    firstAction = P.actions == P.max_actions
    sufficientMinor = P.minor_actions >= 2
    skirmStart = False
    for sk in Skirmishes[0]:
        if (P.birthcity in sk) and (Skirmishes[0][sk] == 2):
            skirmStart = True
    return firstAction * sufficientMinor * skirmStart


def JoinFight(_=None):
    P = lclPlayer()
    city_against = P.PlayerTrack.Quest.quests[3, 6].city_against
    foename = city_against[0].upper() + city_against[1:] + ' Fighter'
    foestyle = cities[city_against]['Combat Style']
    encounter(foename, [30, 50], [foestyle], {'coins': [3, 6]}, enc=0)


def TransportToField(_=None):
    P = lclPlayer()
    skirmishesPossible = set()
    for sk in Skirmishes[0]:
        if (P.birthcity in sk) and (Skirmishes[0][sk] == 2):
            skirmishesPossible = skirmishesPossible.union(sk.difference({P.birthcity}))
    skirmishesPossible = list(skirmishesPossible)
    city_against = np.random.choice(skirmishesPossible) if len(skirmishesPossible) > 0 else skirmishesPossible[0]
    coord, distance, path = P.currenttile.findNearest(city_against)
    mid_i = int(np.median(np.arange(len(path))))
    mid_tile = path[mid_i]
    P.moveto((mid_tile.gridx, mid_tile.gridy), False, True)
    P.PlayerTrack.Quest.quests[3, 6].city_against = city_against
    P.PlayerTrack.Quest.quests[3, 6].action = 1
    JoinFight()


def PresentCraftBooks(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.addItem("crafting book", -3)
    P.PlayerTrack.Quest.update_quest_status((3, 7), 'complete')
    if P.birthcity == 'fodker':
        P.addItem('old fodker cloth', 3)
    else:
        output(f"Village output efficiency increased by 1 for {P.birthcity}!", 'green')
        capital_info[P.birthcity]['efficiency'] += 1
        socket_client.send('[EFFICIENCY]', P.birthcity)
    exitActionLoop()()


def DistributeSand(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[3, 8], 'count'):
        P.PlayerTrack.Quest.quests[3, 8].count = 1
    else:
        P.PlayerTrack.Quest.quests[3, 8].count += 1
    output(f"Delivered {P.PlayerTrack.Quest.quests[3, 8].count} bags of sand so far", 'blue')
    P.addItems('sand', -1)
    if P.PlayerTrack.Quest.quests[3, 8].count >= 10:
        P.PlayerTrack.Quest.update_quest_status((3, 8), 'complete')
        output("New glass making school gives you 3 pieces of glass", 'green')
        P.addItem('glass', 3)
    exitActionLoop('minor')()


def FillLibrary(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 1], 'books_given'):
        P.PlayerTrack.Quest.quests[4, 1].books_given = set()

    def give_book(book, _=None):
        P.PlayerTrack.Quest.quests[4, 1].books_given.add(book)
        output("You have now given: {', '.join(list(P.PlayerTrack.Quest.quests[4, 1].books_given))}", 'blue')
        if len(P.PlayerTrack.Quest.quests[4, 1].books_given) >= 5:
            P.PlayerTrack.Quest.update_quest_status((4, 1), 'complete')
            P.updateSkill("Critical Thinking", 2 if P.skills['Critical Thinking'] < 8 else 1)
        exitActionLoop('minor')()

    actions = {'Cancel': exitActionLoop(amt=0)}
    for item in P.items:
        if ('book' in item) and (item not in P.PlayerTrack.Quest.quests[4, 1].books_given):
            actions[item] = partial(give_book, item)
    actionGrid(actions, False)


def FindWarrior(_=None):
    P = lclPlayer()
    if P.paused:
        return
    city = P.currenttile.tile
    if city not in cities:
        output("You are not in a city!", 'yellow')
        return
    elif city == P.birthcity:
        output("You need to search in cities part from yours!", 'yellow')
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 2], 'cities_searched'):
        P.PlayerTrack.Quest.quests[4, 2].cities_searched = set()
        P.PlayerTrack.Quest.quests[4, 2].cities_found = set()
    if city not in P.PlayerTrack.Quest.quests[4, 2].cities_searched:
        P.PlayerTrack.Quest.quests[4, 2].cities_searched.add(city)
        excavating = P.activateSkill("Excavating")
        r = rbtwn(1, 12, None, excavating, 'Excavating ')
        if r <= excavating:
            P.useSkill("Excavating")
            output(f"You found a {P.birthcity} warrior in {city}! He responds to your call and heads to {P.birthcity}!",
                   'green')
            P.PlayerTrack.Quest.quests[4, 2].cities_found.add(city)
            if len(P.PlayerTrack.Quest.quests[4, 2].cities_found) >= 7:
                P.PlayerTrack.Quest.update_quest_status((4, 2), 'complete')
                output("Your Hit Points and Stability are boosted by 2!", 'green')
                P.boosts[P.attributes["Hit Points"]] += 2
                P.current[P.attributes["Hit Points"]] += 2
                P.boosts[P.attributes["Stability"]] += 2
                P.boosts[P.attributes["Stability"]] += 2
        elif (len(P.PlayerTrack.Quest.quests[4, 2].cities_searched) - len(
                P.PlayerTrack.Quest.quests[4, 2].cities_found)) >= 7:
            output("It is no longer possible for you to find 7 warriors!", 'red')
            P.PlayerTrack.Quest.update_quest_status((4, 2), 'failed')
        else:
            failsleft = 7 - (len(P.PlayerTrack.Quest.quests[4, 2].cities_searched) - len(
                P.PlayerTrack.Quest.quests[4, 2].cities_found))
            output(f"You failed to find a warrior in {city}! You can only fail {failsleft} more times!", 'red')
        exitActionLoop()()
    else:
        output("You have already searched this city!", 'yellow')


def ConvinceMarketLeader(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 3], 'count'):
        P.PlayerTrack.Quest.quests[4, 3].count = 0
    bartering = P.activateSkill("Bartering")
    persuasion = P.activateSkill("Persuasion")
    r = rbtwn(1, 24, None, bartering + persuasion, 'Convincing Market Leader ')
    if r <= (bartering + persuasion):
        P.useSkill("Bartering")
        P.useSkill("Persuasion")
        output("You convinced the market leader to lower his prices!", 'green')
        P.PlayerTrack.Quest.quests[4, 3].count += 1
        if P.PlayerTrack.Quest.quests[4, 3].count >= 6:
            P.PlayerTrack.Quest.update_quest_status((4, 3), 'complete')
            output(f"Market prices reduced by 1 (min=1) in {P.birthcity}!", 'green')
            capital_info[P.birthcity]['discount'] += 1
            socket_client.send('[DISCOUNT]', P.birthcity)
        else:
            output("You still need to convince {6 - P.PlayerTrack.Quest.quests[4, 3].count} more market leaders.",
                   'blue')
    else:
        output("You were unable to convince them this time.", 'yellow')
    exitActionLoop()()


def PursuadeTrader(_=None):
    P = lclPlayer()
    if P.paused:
        return
    persuasion = P.activateSkill("Persuasion")
    r = rbtwn(1, 12, None, persuasion, "Persuasion ")
    if r <= persuasion:
        P.useSkill("Persuasion")
        output(f"Successfully pursuaded trader to start coming to {P.birthcity}!", 'green')
        P.PlayerTrack.Quest.quests[4, 4].count += 1
        if P.PlayerTrack.Quest.quests[4, 4].count >= 3:
            P.PlayerTrack.Quest.update_quest_status((4, 4), 'complete')
            output(f"Traders now have a 1/8 chance of appearing in {P.birthcity}!", 'green')
            capital_info[P.birthcity]['trader allowed'] = True
            socket_client.send('[TRADER ALLOWED]', P.birthcity)
    else:
        output("Unable to convince the trader.", 'yellow')
    exitActionLoop()()


def FindAndPursuadeLeader(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 5], 'count'):
        P.PlayerTrack.Quest.quests[4, 5].count = 0
    if not hasattr(P.PlayerTrack.Quest.quests[4, 5], 'skirmish'):
        P.PlayerTrack.Quest.quests[4, 5].skirmish = {P.currenttile.tile, P.birthcity}
    if {P.currenttile.tile, P.birthcity} != P.PlayerTrack.Quest.quests[4, 5].skirmish:
        output(
            f"You already began trying to reduce tension between {' and '.join(list(P.PlayerTrack.Quest.quests[4, 5].skirmish))}. You must stick to that!",
            'yellow')
        return
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 10, None, excavating, "Excavating ")
    if r <= excavating:
        P.useSkill("Excavating")
        output("You found a leader!")
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 12, None, persuasion, "Persuasion ")
        if r <= persuasion:
            P.useSkill("Persuasion")
            P.PlayerTrack.Quest.quests[4, 5].count += 1
            output(
                f"You convinced the leader to lessen the war effort! {5 - P.PlayerTrack.Quest.quests[4, 5].count} left to go!",
                'blue')
            if P.PlayerTrack.Quest.quests[4, 5].count >= 5:
                P.PlayerTrack.Quest.update_quest_status((4, 5), 'complete')
                skirmish = frozenset([P.birthcity, P.currenttile.tile])
                output("You reduced the tensions between {' and '.join(list(skirmish))} by a factor of 3!", 'green')
                Skirmishes[1][skirmish] += 3
                socket_client.send('[REDUCED TENSION]', [skirmish, 3])
        else:
            output("Failed to persuade the leader.", 'yellow')
    else:
        output("Failed to find a leader.", 'yellow')
    exitActionLoop()()


def StoreTatteredBook(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 6], 'count'):
        P.PlayerTrack.Quest.quests[4, 6].count = 1
    else:
        P.PlayerTrack.Quest.quests[4, 6].count += 1
    output(
        f"You found an old book containing long lost history! You have found {P.PlayerTrack.Quest.quests[4, 6].count} so far.",
        'blue')
    exitActionLoop()()


def DeliverTatteredBooks(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((4, 6), 'complete')
    coins = min([40, 10 * P.PlayerTrack.Quest.quests[4, 6].count])
    output(f"You received {coins} coins!", 'green')
    P.coins += coins
    exitActionLoop()()


def IncreaseCapacity(city, amt, _=None):
    P = lclPlayer()
    capital_info[city]['capacity'] += amt
    if P.home[city]:
        P.max_capacity += amt


def HomeRenovations(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 7], 'remaining'):
        P.PlayerTrack.Quest.quests[4, 7].remaining = {'bark': 5, 'clay': 3, 'glass': 3, 'leather': 2}

    def distribute(item, _=None):
        P.PlayerTrack.Quest.quests[4, 7].remaining[item] -= 1
        allZero = True
        for val in P.PlayerTrack.Quest.quests[4, 7].remaining.values():
            if val > 0:
                allZero = False
        if allZero:
            P.PlayerTrack.Quest.update_quest_status((4, 7), 'complete')
            output(f"Homes in {P.birthcity} have increased capacity of 3!", 'green')
            IncreaseCapacity(P.birthcity, 3)
            socket_client.send('[CAPACITY]', [P.birthcity, 3])
        exitActionLoop('minor')()

    actions = {'Cancel': exitActionLoop(amt=0)}
    for item in P.items:
        if (item in P.PlayerTrack.Quest.quests[4, 7].remaining) and (
                P.PlayerTrack.Quest.quests[4, 7].remaining[item] > 0):
            actions[item[0].upper() + item[1:]] = partial(distribute, item)
    actionGrid(actions, False)


def ConvinceWarriorsToRaid(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if not hasattr(P.PlayerTrack.Quest.quests[4, 8], 'cities'):
        P.PlayerTrack.Quest.quests[4, 8].cities = {}
    elif (len(P.PlayerTrack.Quest.quests[4, 8]) >= 4) and (
            P.currenttile.tile not in P.PlayerTrack.Quest.quests[4, 8].cities):
        output("You have already started looking in other cities!", 'yellow')
        return
    if (P.birthcity not in P.PlayerTrack.Quest.quests[4, 8].cities) and (P.currenttile.tile != P.birthcity):
        output("You need to start with your city first!", 'yellow')
        return
    elif (P.currenttile.tile in P.PlayerTrack.Quest.quests[4, 8].cities) and (
            P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] >= 3):
        output("You have already found and convinced 3 warriors in this city", 'yellow')
        return
    elif P.currenttile.tile in Skirmishes[2][P.birthcity]:
        output("You cannot convince any warriors in this city due to tensions!", 'yellow')
        return
    elif P.currenttile.tile not in P.PlayerTrack.Quest.quests[4, 8].cities:
        # Initiate the count for the city
        P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] = 0
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 12, None, excavating, 'Excavating ')
    if r <= excavating:
        output("You found a warrior")
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 10, None, persuasion, 'Persuasion ')
        if r <= persuasion:
            P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] += 1
            if P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] >= 3:
                output(
                    f"You have convinced 3 warriors in {P.currenttile.tile} to raid the caves. They set out and do so!",
                    'green')
                if len(P.PlayerTrack.Quest.quests[4, 8].cities) >= 4:
                    allFinished = True
                    for val in P.PlayerTrack.Quest.quests[4, 8].cities.values():
                        if val < 3:
                            allFinished = False
                    if allFinished:
                        P.PlayerTrack.Quest.update_quest_status((4, 8), 'completed')
            else:
                output(
                    "You persuade them to raid the caves once all three in the {P.currentile.tile} are ready. You have convinced {P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile]} so far in {P.currentile.tile}!",
                    'blue')
        else:
            output("Unable to persuade the warrior", 'yellow')
    else:
        output("Unable to find any warrior", 'yellow')
    exitActionLoop()()


def ConvinceWarriorToJoin(_=None):
    P = lclPlayer()
    if P.paused:
        return
    # Assumption: Logic to determine whether finding warrior in this city is valid is handled somewhere else
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 12, None, excavating, 'Excavating ')
    if r <= excavating:
        lvl = max(rbtwn(40, 80, max([1, excavating // 3])))
        output(f"You found a warrior lvl {lvl}")
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 6 + round((80 - lvl) / 6), None, persuasion, 'Persuasion ')
        if r <= persuasion:
            output("You convinced the warrior to join you for 6 rounds! Any other warriors leave you group.", 'green')
            P.has_warrior = 6
            warriorstats = npc_stats(lvl)
            P.group["Warrior"] = warriorstats
        else:
            output("Unable to convince them to join you.", 'yellow')
    else:
        output("Unable to find any warriors.", 'yellow')
    exitActionLoop()()


def GoldStart(_=None):
    P = lclPlayer()
    P.PlayerTrack.Quest.quests[5, 1].has_gold = False


def SmithGold(_=None):
    P = lclPlayer()
    if P.paused:
        return
    persuasion = P.activateSkill("Persuasion")
    r = rbtwn(1, 14, None, persuasion, 'Persuasion ')
    if r <= persuasion:
        output(
            "You were able to convince the smith to make the dagger! Mayor rewards you with 20 coins and your impact stability reduces by 1.",
            'green')
        P.PlayerTrack.Quest.update_quest_status((5, 1), 'complete')
        P.coins += 20
        P.stability_impact += 1
    else:
        output("Unable to convince the smith", 'yellow')
    exitActionLoop()()


def PlaceFriend(_=None):
    P = lclPlayer()
    allPos = []
    for tileType in ['randoms', 'ruins', 'battle1', 'battle2', 'wilderness']:
        allPos += positions[tileType]
    randPos = np.random.choice(allPos)
    P.PlayerTrack.Quest.quests[5, 2].coord = randPos
    P.PlayerTrack.Quest.quests[5, 2].has_friend = False


def ShowFriend(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((5, 2), 'complete')
    output("You gain 20 coins and max actions per round increases by 1!", 'green')
    P.coins += 20
    P.max_actions += 1
    exitActionLoop('minor')()


def KilledMonster(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((5, 3), 'complete')
    P.coins += 30
    for city in Skirmishes[2][P.birthcity]:
        skirmish = frozenset([city, P.birthcity])
        Skirmishes[1][skirmish] += 1
        output("You reduced the tensions between {' and '.join(list(skirmish))} by a factor of 1!", 'green')
        socket_client.send('[REDUCED TENSION]', [skirmish, 1])
    exitActionLoop('minor')()


def ShowRareOre(_=None):
    P = lclPlayer()
    if P.paused:
        return
    rares = {'shinopsis', 'ebony', 'astatine', 'promethium'}
    raresLeft = rares.intersection(P.items)

    def giveRare(raresSelected, raresLeft, _=None):
        if len(raresSelected) >= 2:
            P.PlayerTrack.Quest.update_quest_status((5, 4), 'complete')
            for rare in raresSelected:
                atr = f'Def-{ore_properties[rare][0]}'
                P.boosts[P.attributes[atr]] += 2
                P.current[P.attributes[atr]] += 2
                output(f"{atr} Boosted by 2!", 'green')
            exitActionLoop()()
        else:
            actions = {'Cancel': exitActionLoop(amt=0)}
            for rare in raresLeft:
                possibleSelect = raresSelected.union({rare})
                possibleLeft = raresLeft.difference({rare})
                actions[rare] = partial(giveRare, possibleSelect, possibleLeft)
            actionGrid(actions, False)

    giveRare(set(), raresLeft)


def ConvinceBarterMaster(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if P.coins < 40:
        output("You lack the ability to pay him!", 'yellow')
        return
    persuasion = P.activateSkill("Persuasion")
    r = rbtwn(1, 20, None, persuasion, 'Persuasion ')
    if r <= persuasion:
        P.useSkill("Persuasion", 2, 7)
        output(
            "You convinced the Bartering Master to teach the mayor his knowledge! You pay him 40 coins. Your max minor actions increase by 4!",
            'green')
        P.coins -= 40
        P.PlayerTrack.Quest.update_quest_status((5, 5), 'complete')
        P.max_minor_actions += 4
    else:
        output("Unable to convince the Bartering Master", 'yellow')
    exitActionLoop()()


def HighlightMammothTile(_=None):
    P = lclPlayer()
    P.parentBoard.gridtiles[P.PlayerTrack.Quest.quests[1, 8].coord_found].color = (1, 1.5, 1, 1)
    output("In case you forgot, the tile you found the baby mammoth is tinted green")


def KilledMammoth(_=None):
    P = lclPlayer()
    if P.paused:
        return
    P.PlayerTrack.Quest.update_quest_status((5, 6), 'complete')
    output("You are awarded 3 gems and 3 diamonds!", 'green')
    P.addItem('gems', 3)
    P.addItem('diamond', 3)
    exitActionLoop('minor')()


def AskMasterStealthForMayor(_=None):
    P = lclPlayer()
    if P.paused:
        return
    persuasion = P.activateSkill("Persuasion")
    cunning = P.current[P.attributes["Cunning"]]
    critthink = P.current[P.attributes["Critical Thinking"]]
    r = rbtwn(1, 36, None, (persuasion + cunning + critthink), 'Convicing ')
    if r <= (persuasion + cunning + critthink):
        output("You convinced the Stealth Master to teach the mayor! You get 2 stealth books and 1 free lesson to use.",
               'green')
        P.PlayerTrack.Quest.update_quest_status((5, 7), 'completed')
        P.addItem("stealth book", 2)
        P.PlayerTrack.Quest.quests[5, 7].used_lesson = False
    else:
        output("You failed to convince the master.", 'yellow')
        if not hasattr(P.PlayerTrack.Quest.quests[5, 7], 'count'):
            P.PlayerTrack.Quest.quests[5, 7].count = 1
        else:
            P.PlayerTrack.Quest.quests[5, 7].count += 1
            if P.PlayerTrack.Quest.quests[5, 7].count >= 2:
                P.PlayerTrack.Quest.update_quest_status((5, 7), 'failed')
    exitActionLoop()()


def GiftPerfectCraft(_=None):
    P = lclPlayer()
    if P.paused:
        return
    perfectCraft = {'gems', 'rubber', 'glass', 'ceramic', 'leather', 'scales', 'beads'}
    gifting = False

    def giftCraft(space):
        P.PlayerTrack.craftingTable.rmv_craft(space)
        P.PlayerTrack.Quest.update_quest_status((5, 8), 'completed')
        output(f"You can now train in {P.birthcity} for free with Adept trainers and half price at Master trainers!",
               'green')
        P.training_discount = True

    for space in [0, 1]:
        if P.PlayerTrack.craftingTable.getItems(space) == perfectCraft:
            giftCraft(space)
            gifting = True
            break
    if not gifting:
        output(f"You do not possess the perfect craft of: {', '.join(list(perfectCraft))}", 'yellow')
    exitActionLoop('minor')()

quest_req = {(1, 3): 'self.playerTrack.player.actions == self.playerTrack.player.max_actions',
             (1, 8): "'fruit' in self.playerTrack.player.items",
             (2, 3): "self.quests[2, 6].status != 'started'",
             (2, 5): "self.quests[1, 6].status == 'complete'",
             (2, 6): "self.quests[2, 3].status != 'started'",
             (2, 7): "self.quests[1, 4].status == 'complete'",
             (3, 4): "self.playerTrack.player.Combat >= 40",
             (3, 6): "StartofSkirmish()", # [INCOMPLETE] Figure out if beginning of skirmish for a city -- then choose that city
             (3, 8): "self.quests[1, 7].status == 'complete'",
             (4, 1): "self.quests[2, 5].status == 'complete'",
             (4, 7): "(self.quests[1, 2].status == 'complete') and (self.playerTrack.player.skills['Crafting'] >= 6)",
             (5, 5): "self.playerTrack.player.coins >= 40",
             (5, 6): "self.quests[1, 8].status == 'complete'",
             (5, 7): "self.quests[2, 8].status == 'complete'",
             (5, 8): "self.playerTrack.player.skills['Crafting'] >= 12"}

quest_activate_response = {(2, 3): BeginProtection,
                           (2, 4): MotherSerpent,
                           (2, 5): LibrariansSecret,
                           (2, 6): TheLetter,
                           (3, 2): FitnessTraining,
                           (3, 6): TransportToField,
                           (5, 1): GoldStart,
                           (5, 2): PlaceFriend,
                           (5, 6): HighlightMammothTile}
city_quest_actions = {(1, 1): ["Find Pet", "True", FindPet],
                      (1, 2): ["Clean House", "True", CleanHome],
                      (1, 3): ["Gaurd Home", "True", GaurdHome],
                      (1, 4): ["Gift Craft", "(self.playerTrack.craftingTable.space_items[0] > 1) or (self.playerTrack.craftingTable.space_items[1] > 1)", OfferCraft],
                      (1, 5): ["Spare with Boy", "True", SpareWithBoy],
                      (1, 6): ["Gift Book", "checkBook()", OfferBook],
                      (1, 7): ["Gift Sand", "'sand' in self.playerTrack.player.items", OfferSand],
                      (1, 8): ["Drop Baby Mammoth at Zoo", "hasattr(self.quests[1, 8], 'has_mammoth') and self.quests[1, 8].has_mammoth", ZooKeeper],
                      (2, 1): ["Apply Cubes", "'cooling cubes' in self.playerTrack.player.items", ApplyCubes],
                      (2, 2): ["Wait for Robber", 'True', WaitforRobber],
                      (2, 5): ["Give Book", 'self.quests[2, 5].has_book', HandOverBook],
                      (2, 7): ["Give Ores", "{'aluminum', 'nickel', 'tantalum', 'kevlium'}.issubset(self.playerTrack.player.items)", GiveOresToSmith],
                      (2, 8): ["Give Stealth Book", "hasattr(self.quests[2, 8], 'has_book') and self.quests[2, 8].has_book", PresentStealthBook],
                      (3, 1): ["Distribute Food", "True", FeedThePoor],
                      (3, 3): ["Study with Monk", "hasattr(self.quests[3, 3], 'convinced_monk')", LearnFromMonk],
                      (3, 4): ["Search for Monster", "self.playerTrack.player.Combat >= 40", SearchForMonster],
                      (3, 5): ["Teach Fishing", 'True', TeachFishing],
                      (3, 7): ["Gift Craft Books", "('crafting book' in self.playerTrack.player.items) and (self.playerTrack.player.items['crafting book'] >= 3)", PresentCraftBooks],
                      (3, 8): ["Deliver Sand", "'sand' in self.playerTrack.player.items", DistributeSand],
                      (4, 1): ["Fill Library", "True", FillLibrary],
                      (4, 3): ["Convince a Market Leader", "True", ConvinceMarketLeader],
                      (4, 6): ["Deliver Historic Books", "hasattr(self.playerTrack.Quest.quests[4, 6], 'count') and (self.playerTrack.Quest.quests[4, 6].count >=2)", DeliverTatteredBooks],
                      (4, 7): ["Deliver Material", "True", HomeRenovations],
                      (5, 1): ["Smith Gold", "self.playerTrack.Quest.quests[5, 1].has_gold", SmithGold],
                      (5, 2): ["Show Friend", "self.playerTrack.Quest.quests[5, 2].has_friend", ShowFriend],
                      (5, 3): ["Claim Reward", "hasattr(self.playerTrack.Quest.quests[5, 3], 'killed')", KilledMonster],
                      (5, 4): ["Show Rare Ore", "len({'shinopsis', 'ebony', 'astatine', 'promethium'}.intersection(self.playerTrack.player.items))>=2", ShowRareOre],
                      (5, 6): ["Claim Reward", "hasattr(self.playerTrack.Quest.quests[5, 6], 'killed')", KilledMammoth],
                      (5, 8): ["Gift Perfect Craft", "({'gems', 'rubber', 'glass', 'ceramic', 'leather', 'scales', 'beads'} == self.playerTrack.craftingTable.getItems(0)) or ({'gems', 'rubber', 'glass', 'ceramic', 'leather', 'scales', 'beads'} == self.playerTrack.craftingTable.getItems(1))", GiftPerfectCraft]}


class Quest:
    def __init__(self, playerTrack):
        self.playerTrack = playerTrack
        qgrid = GridLayout(cols=1)
        self.Q = pd.read_csv('data\\QuestTable.csv')
        self.questDisplay = Button(text='', height=Window.size[1]*0.15,size_hint_y=None,color=(0,0,0,1),markup=True,background_color=(1,1,1,0))
        self.questDisplay.text_size = self.questDisplay.size
        self.questDisplay.bind(size=self.update_bkgSize)
        self.quests, data, self.stage_completion = {}, [], {i: 0 for i in range(1, 6)}
        for mission in range(1, 9):
            datarow = [Button(text=str(mission), disabled=True, background_disabled_normal='', color=(0,0,0,1))]
            for stage in range(1, 6):
                row = self.Q[(self.Q['Stage']==stage)&(self.Q['Mission']==mission)]
                msg = f"{row['Name'].iloc[0]}\nRequirement: " + str(row['Requirements'].iloc[0]) + ' | Failable: ' + str(row['Failable'].iloc[0]) + ' | Reward: ' + str(row['Rewards'].iloc[0]) + '\nProcedure: '+ str(row['Procedure'].iloc[0])
                B = HoverButton(self.questDisplay, msg, text=row['Name'].iloc[0], disabled=False if (stage==1) and self.req_met((stage, mission)) else True)
                B.mission = mission
                B.stage = stage
                B.status = 'not started'
                datarow.append(B)
                self.quests[stage, mission] = B
                B.bind(on_press=self.activate)
            data.append(datarow)
        self.QuestTable = Table(['Mission','Stage 1\nCommon Folk', 'Stage 2\nNoblemen', 'Stage 3\nDistrict Leaders', 'Stage 4\nCity Counsel', 'Stage 5\nMayor'], data, header_color=(50, 50, 50), header_as_buttons=True)
        qgrid.add_widget(self.QuestTable)
        qgrid.add_widget(self.questDisplay)
        screen = Screen(name = 'Reputation')
        screen.add_widget(qgrid)
        self.playerTrack.track_screen.add_widget(screen)
    def update_bkgSize(self, instance, value):
        self.questDisplay.text_size = self.questDisplay.size
    def update_reqs(self):
        for B in self.quests.values():
            if B.status == 'not started':
                B.disabled = False if self.req_met((B.stage, B.mission)) else True
                self.update_citypage((B.stage, B.mission))
                if B.disabled: B.color = (1, 1, 1, 1)
    def saveTable(self):
        variableExclusions = {'_trigger_texture', '_context', '_disabled_value', '_disabled_count', 'canvas', '_proxy_ref', '_label', '_ButtonBehavior__state_event', '_ButtonBehavior__touch_time', 'display', 'message', 'mission', 'stage'}
        for mission in range(1, 9):
            for stage in range(1, 6):
                B = self.quests[stage, mission]
                self.playerTrack.player.reputation[stage-1, mission-1] = {field: B.__dict__[field] for field in set(B.__dict__).difference(variableExclusions)}
    def loadTable(self):
        for mission in range(1, 9):
            for stage in range(1, 6):
                if self.playerTrack.player.reputation[stage-1, mission-1]['status'] != 'not started':
                    for field, value in self.playerTrack.player.reputation[stage-1, mission-1].items():
                        if field == 'status':
                            self.update_quest_status((stage, mission), value, False)
                        else:
                            setattr(self.quests[stage, mission], field, value)
    def req_met(self, quest):
        init_req = eval(quest_req[quest]) if quest in quest_req else True
        inCity = self.playerTrack.player.currenttile.tile == self.playerTrack.player.birthcity
        return init_req * inCity if (quest[0] == 1) else init_req * (self.stage_completion[quest[0]-1] >= 4) * inCity

    def update_citypage(self, quest):
        if self.playerTrack.player.parentBoard.game_page.city_page is not None:
            if self.quests[quest].disabled:
                self.playerTrack.player.parentBoard.game_page.city_page.disable_active_quest(quest[0], quest[1])
            else:
                self.playerTrack.player.parentBoard.game_page.city_page.enable_quest(quest[0], quest[1])
    def update_quest_status(self, quest, new_status=None, verbose=True):
        if (new_status == 'started'):
            self.quests[quest].disabled = True
            self.quests[quest].background_color = (1.5, 1.5, 0.5, 1.5)
            self.quests[quest].color = (0.6, 0.6, 0.6, 1)
        elif (new_status == 'not started'):
            self.quests[quest].disabled = not self.req_met(quest)
            self.quests[quest].background_color = (1, 1, 1, 1)
            self.quests[quest].color = (1, 1, 1, 1)
            if verbose: output(f"Mission '{self.quests[quest].text}' must be restarted!", 'red')
        elif (new_status == 'failed'):
            self.quests[quest].disabled = True
            if verbose: output(f"You failed the mission '{self.quests[quest].text}'!", 'red')
            self.quests[quest].background_color = (3, 0, 0, 1.5)
            self.quests[quest].color = (0.6, 0.6, 0.6, 1)
        elif (new_status == 'complete'):
            self.quests[quest].disabled = True
            if verbose: output(f"You completed the mission '{self.quests[quest].text}'! Reputation increases by {self.quests[quest].stage}!", 'green')
            self.quests[quest].background_color = (0, 3, 0, 1.5)
            self.quests[quest].color = (0.6, 0.6, 0.6, 1)
            self.playerTrack.player.Reputation += self.quests[quest].stage
            self.playerTrack.player.parentBoard.checkCityUnlocks() # Remove lock icons if appropriate
            self.update_reqs()
            self.playerTrack.player.updateTotalVP(self.quests[quest].stage)
            self.playerTrack.reputationTab.text = f"Reputation: [color={self.playerTrack.hclr}]{self.playerTrack.player.Reputation}[/color]"
            self.stage_completion[self.quests[quest].stage] += 1
            for city in city_info:
                if self.playerTrack.player.Reputation >= city_info[city]['entry']:
                    if not self.playerTrack.player.entry_allowed[city]:
                        output(f"You are now allowed to enter {city}!", 'green')
                        output(f"Get {2*city_info[city]['entry']} to buy a market or home in {city}.", 'blue')
                    self.playerTrack.player.entry_allowed[city] = True
                if self.playerTrack.player.Reputation >= (2*city_info[city]['entry']):
                    if not self.playerTrack.player.market_allowed[city]:
                        output(f"You are now allowed to buy a market or home in {city}!", 'green')
                        output(f"Get {3*city_info[city]['entry']} or purchase home to train in {city}.", 'blue')
                    self.playerTrack.player.market_allowed[city] = True
                elif (self.playerTrack.player.Reputation >= (3*city_info[city]['entry'])) and (not self.playerTrack.player.training_allowed[city]):
                    if not self.playerTrack.player.training_allowed[city]:
                        output(f"You are now allowed to train in {city}!", 'green')
                    self.playerTrack.player.training_allowed[city] = True
        else:
            # Each statement was failed so assume that the status should not be changed
            return
        self.quests[quest].status = new_status
        self.update_citypage(quest)
    def activate(self, instance):
        if self.playerTrack.player.paused:
            return
        quest = (instance.stage, instance.mission)
        if not self.req_met(quest):
            output("You do not meet the requirements for starting this quest!", 'yellow')
            return
        self.update_quest_status(quest, 'started')
        if quest in quest_activate_response:
            quest_activate_response[quest]()
        exitActionLoop('minor')()
    def add_active_city_actions(self, actions):
        for B in self.quests.values():
            if B.status == 'started':
                quest = (B.stage, B.mission)
                if (quest in city_quest_actions) and eval(city_quest_actions[quest][1]):
                    actions['*b|'+city_quest_actions[quest][0]] = city_quest_actions[quest][2]
        return actions

#%% Player Track: Grid
def level_up_color(level, min_lvl=1, max_lvl=12):
    if level < min_lvl:
        return (0, 0, 0, 1)
    if level > max_lvl:
        level = max_lvl
    rgb = skcolor.lab2rgb((44.12, (178/(max_lvl-1))*(level-1)-50, 48.2))
    return (rgb[0], rgb[1], rgb[2], 1)

def toReadableTime(secondsElapsed, roundTo=2):
    if secondsElapsed is None:
        return '-'
    if secondsElapsed < 0:
        secondsElapsed = -secondsElapsed
    if secondsElapsed < 60:
        return f"{np.round(secondsElapsed, roundTo)} seconds"
    elif secondsElapsed < 3600:
        return f"{np.round(secondsElapsed / 60, roundTo)} minutes"
    elif secondsElapsed < 86400:
        return f"{np.round(secondsElapsed / 3600, roundTo)} hours"
    return f"{np.round(secondsElapsed / 86400, roundTo)} days"

class PlayerTrack(GridLayout):
    def __init__(self, player, **kwargs):
        super().__init__(**kwargs)
        self.player = player
        self.cols=1
        self.hclr = essf.get_hexcolor((255, 85, 0))
        self.track_screen = ScreenManager()
        # Combat Screen
        combatGrid = GridLayout(cols=1)
        self.combatDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0,0,0,1),markup=True,background_color=(1,1,1,0))
        self.Combat = Table(header=['Attribute','Base','Boost','Total'], data=self.get_Combat(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
        self.Combat.update_header('Total', f'Total: {self.get_total_combat()}')
        self.armoryTable = ArmoryTable()
        combatGrid.add_widget(self.Combat)
        combatGrid.add_widget(self.armoryTable)
        combatGrid.add_widget(self.combatDisplay)
        screen = Screen(name='Combat')
        screen.add_widget(combatGrid)
        self.track_screen.add_widget(screen)
        # Knowledge Screen
        knowledgeGrid = GridLayout(cols=1)
        self.knowledgeDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0,0,0,1),markup=True,background_color=(1,1,1,0))
        self.Knowledge = Table(header=['Skill', 'Level', 'XP'], data=self.get_Knowledge(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
        knowledgeGrid.add_widget(self.Knowledge)
        knowledgeGrid.add_widget(self.knowledgeDisplay)
        screen = Screen(name='Knowledge')
        screen.add_widget(knowledgeGrid)
        self.track_screen.add_widget(screen)
        # Capital Screen
        capGrid = GridLayout(cols=1)
        self.capitalDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0, 0, 0, 1),markup=True,background_color=(1,1,1,0))
        self.Capital = Table(header=['City','Home','Market','Bank','Villages Invested','Waiting'], data=self.get_Capital(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True)
        capGrid.add_widget(self.Capital)
        capGrid.add_widget(self.capitalDisplay)
        screen = Screen(name='Capital')
        screen.add_widget(capGrid)
        self.track_screen.add_widget(screen)
        # Reputation Screen
        self.Quest = Quest(self)
        # Fellowship Screen
        fellowshipGrid = GridLayout(cols=1)
        screen = Screen(name='Fellowship')

        screen.add_widget(fellowshipGrid)
        self.track_screen.add_widget(screen)
        # Titles Screen
        titleGrid = GridLayout(cols=1)
        self.titleDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0, 0, 0, 1),markup=True,background_color=(1,1,1,0))
        self.Titles = Table(header=['Title', 'Category', 'VP', 'Minimum Req', 'Your Record', 'Highest Record'], data=self.get_Titles(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
        titleGrid.add_widget(self.Titles)
        titleGrid.add_widget(self.titleDisplay)
        screen = Screen(name='Titles')
        screen.add_widget(titleGrid)
        self.track_screen.add_widget(screen)
        # Item Screen
        itemGrid = GridLayout(cols=1)
        self.Items = Table(header=['Item', 'Quantity', 'Sell', 'Use'], data=self.get_Items(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True)
        self.craftingTable = CraftingTable()
        itemGrid.add_widget(self.Items)
        itemGrid.add_widget(self.craftingTable)
        screen = Screen(name='Items')
        screen.add_widget(itemGrid)
        self.track_screen.add_widget(screen)
        # Add header tabs
        def changeTab(tabName, tabButton):
            self.track_screen.current = tabName
            for tB in self.tabs.children:
                tB.disabled = True if tB == tabButton else False
                tB.color = (0, 0, 0, 1) if tB == tabButton else (1, 1, 1, 1)
        self.tabs = GridLayout(cols=7, height=0.07*Window.size[1], size_hint_y=None)
        self.tab_color = (0.1, 0.6, 0.5, 1)
        self.combatTab = Button(text=f"Combat: [color={self.hclr}]{player.Combat}[/color]",markup=True,background_color=self.tab_color)
        self.combatTab.bind(on_press=partial(changeTab, 'Combat'))
        self.knowledgeTab = Button(text=f'Knowledge: [color={self.hclr}]{player.Knowledge}[/color]',markup=True,background_color=self.tab_color)
        self.knowledgeTab.bind(on_press=partial(changeTab, 'Knowledge'))
        self.capitalTab = Button(text=f'Capital: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.capitalTab.bind(on_press=partial(changeTab, "Capital"))
        self.reputationTab = Button(text=f'Reputation: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.reputationTab.bind(on_press=partial(changeTab, "Reputation"))
        self.fellowshipTab = Button(text=f'Fellowship: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.fellowshipTab.bind(on_press=partial(changeTab, "Fellowship"))
        self.titlesTab = Button(text=f'Titles: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.titlesTab.bind(on_press=partial(changeTab, "Titles"))
        self.itemsTab = Button(text='Items: 0/3',markup=True,background_color=self.tab_color)
        self.itemsTab.bind(on_press=partial(changeTab, 'Items'))
        self.tabs.add_widget(self.combatTab)
        self.tabs.add_widget(self.knowledgeTab)
        self.tabs.add_widget(self.capitalTab)
        self.tabs.add_widget(self.reputationTab)
        self.tabs.add_widget(self.fellowshipTab)
        self.tabs.add_widget(self.titlesTab)
        self.tabs.add_widget(self.itemsTab)
        # Add widgets in order
        self.add_widget(self.tabs)
        self.add_widget(self.track_screen)
    def get_Combat(self):
        data = []
        for i in range(len(self.player.atrorder)):
            atr = self.player.atrorder[i]
            if atr in {'Attack', 'Technique'}:
                adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([('[color=00f03e]' if (city not in cities) or self.player.training_allowed[city] else '[color=b50400]')+city[0].upper()+city[1:]+'[/color]' for city in adept_loc[self.player.combatstyle]])
                mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([('[color=00f03e]' if (city not in cities) or self.player.training_allowed[city] else '[color=b50400]')+city[0].upper()+city[1:]+'[/color]' for city in master_loc[self.player.combatstyle]])
            else:
                adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in adept_loc[atr]])
                mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in master_loc[atr]])
            if (atr[:4] == 'Def-') or (atr == 'Attack'):
                cmbtstyle = atr[4:] if atr[:4] == 'Def-' else self.player.combatstyle
                boosts = '\n[color=d66000]Boosts[/color]: '+', '.join([f'{i+1} - {inverse_ore_properties[cmbtstyle][i]}' for i in range(len(inverse_ore_properties[cmbtstyle]))])
            else:
                boosts = ''
            msg = adpt+' | '+mstr+boosts
            current = {'text':str(self.player.current[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}
            if len(self.player.group) > 0:
                group_disp = []
                for warrior, lvls in self.player.group.items():
                    group_disp.append(f'{warrior}: {lvls[i]}')
                current['hover'] = (self.combatDisplay, ', '.join(group_disp))
            data.append([{'text':atr,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1),'hover':(self.combatDisplay, msg)},
                         {'text':str(self.player.combat[i]),'disabled':True,'background_color':(1,1,1,0),'color':level_up_color(self.player.combat[i])},
                         {'text':str(self.player.boosts[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         current])
        return data
    def get_total_combat(self):
        return int(np.sum(self.player.current))
    def get_Knowledge(self):
        data = []
        for skill in self.player.skills:
            adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([('[color=00f03e]' if (city not in cities) or self.player.training_allowed[city] else '[color=b50400]')+city[0].upper()+city[1:]+'[/color]' for city in adept_loc[skill]])
            mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([('[color=00f03e]' if (city not in cities) or self.player.training_allowed[city] else '[color=b50400]')+city[0].upper()+city[1:]+'[/color]' for city in master_loc[skill]])
            msg = adpt+' | '+mstr
            xp = f'{self.player.xps[skill]} / {3 + self.player.skills[skill]}' if self.player.skills[skill] < 8 else f'{self.player.xps[skill]} / ---'
            data.append([{'text':skill,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1),'hover':(self.knowledgeDisplay, msg)},
                         {'text':str(self.player.skills[skill]),'disabled':True,'background_color':(1,1,1,0),'color':level_up_color(self.player.skills[skill])},
                         {'text':xp,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}])
        return data
    def skirmDisplay(self, city, _=None):
        data = {'text':city[0].upper()+city[1:],'color':(0,0,0,1), 'disabled':True}
        fighting = set()
        for sk in Skirmishes[0]:
            if city in sk:
                fighting = fighting.union(sk.difference({city}))
        msg = '[color=009e33]Not skirmishing![/color]' if len(fighting) == 0 else '[color=b50400]In skirmish with[/color]: '+', '.join([city[0].upper()+city[1:] for city in fighting])
        reputation = "\n" if self.player.entry_allowed[city] else f"\nEntry Reputation: [color={self.hclr}]{city_info[city]['entry']}[/color] | "
        items_allowed = 'Sold' if self.player.market_allowed[city] else '[color=b50400]Locked[/color]'
        msg += f"{reputation}Items {items_allowed}: "+', '.join(sorted(city_info[city]['sell']))
        data['hover'] = (self.capitalDisplay, msg)
        data['background_color'] = (1,1,1,0) if len(fighting) == 0 else (1, 0, 0, 0.3)
        return data
    def homeDisplay(self, city, _=None):
        data = {}
        msg = f' Capital: [color={self.hclr}]{capital_info[city]["home_cap"]}[/color], Capacity: {capital_info[city]["capacity"]}'
        if self.player.homes[city]:
            msg = 'Purchased!'+ msg
            data['disabled'] = True
            data['text'] = 'Owned!'
            data['background_color'] = (1, 10, 1, 0.8)
            data['color'] = (0, 0, 0, 1)
        elif not self.player.market_allowed[city]:
            msg = f"You need at least {2*city_info[city]['entry']} Reputation to purchase the home!" + msg
            data['disabled'] = True
            data['text'] = f"{2*city_info[city]['entry']} Reputation\nRequired"
        else:
            msg = f'Cost: [color=cfb53b]{capital_info[city]["home"]}[/color]' + msg
            data['text'] = 'Buy'
            if self.player.currenttile.tile != city:
                data['disabled'] = True
            else:
                data['background_color'] = (2, 2, 0.2, 1)
                data['func'] = partial(self.player.purchase, 'home', city, None)
        data['hover'] = (self.capitalDisplay, city[0].upper()+city[1:]+' Home | '+msg)
        return data
    def marketDisplay(self, city, _=None):
        data = {}
        msg = f' Capital: [color={self.hclr}]{capital_info[city]["market_cap"]}[/color], Income: [color=00d900]{capital_info[city]["return"]}[/color]'
        if self.player.markets[city]:
            if self.player.workers[city]:
                msg = 'Owned and Automated!' + msg
                data['disabled'] = True
                data['text'] = 'Automated'
                data['background_color'] = (1, 10, 1, 0.8)
                data['color'] = (0, 0, 0, 1)
            else:
                msg = 'Owned!' + msg
                data['text'] = 'Automate'
                if self.player.currenttile.tile != city:
                    data['disabled'] = True
                else:
                    data['background_color'] = (2, 2, 0.2, 1)
                    data['func'] = partial(self.player.get_worker, city)
        elif not self.player.market_allowed[city]:
            msg = f"You need at least {2*city_info[city]['entry']} Reputation to purchase the market!" + msg
            data['text'] = f"{2*city_info[city]['entry']} Reputation\nRequired"
            data['disabled'] = True
        else:
            msg = f'Cost: [color=cfb53b]{capital_info[city]["market"]}[/color]' + msg
            data['text'] = 'Buy'
            if self.player.currenttile.tile != city:
                data['disabled'] = True
            else:
                data['background_color'] = (2, 2, 0.2, 1)
                data['func'] = partial(self.player.purchase, 'market', city, None)
        tensions = []
        for S in Skirmishes[1]:
            if city in S:
                other_city = list(S.difference(set([city]))).pop()
                tensions.append(other_city[0].upper()+other_city[1:]+f' ([color=ff3718]{Skirmishes[1][S]}[/color])')
        data['hover'] = (self.capitalDisplay, city[0].upper()+city[1:]+' Market | '+msg+', Tensions: '+', '.join(tensions))
        return data
    def villageDisplay(self, city, _=None):
        data = {}
        cost = capital_info[city]['invest']
        output = None if cost is None else round((12-cost) + (6 - (12-cost))/1.9) - capital_info[city]['efficiency']
        msg = f'Capital: [color={self.hclr}]1ea[/color], Cost: [color=cfb53b]{cost}[/color], Output Speed: [color=00d900]{output}[/color], '+', '.join([f'V{v[-1]}: '+('[color=b50400]Not Invested[/color]' if v not in self.player.villages[city] else f'[color=bf004b]{self.player.villages[city][v][0]}[/color] ({self.player.villages[city][v][1]})') for v in city_villages[city]])
        if (self.player.currenttile.tile == 'village3') and ('village3' not in self.player.villages[city]) and (city in self.player.currenttile.neighbortiles):
            data['text'] = f'Invest: [color=ffff75]-{capital_info[city]["invest"]}[/color]'
            data['func'] = partial(self.player.invest)
            data['markup'] = True
        else:
            data['disabled'] = True
            data['text']='-' if city=='fodker' else f'{len(self.player.villages[city])} / {len(city_villages[city])}'
            data['background_color'] = (1, 1, 1, 0)
            data['color'] = (0, 0, 0, 1)
        data['hover'] = (self.capitalDisplay, city[0].upper()+city[1:]+' Villages | '+('None' if city=='fodker' else msg))
        return data
    def villageAwaitingDisplay(self, city, _=None):
        data = {}
        data['disabled'] = True
        data['background_color'] = (1, 1, 1, 0)
        data['color'] = (0, 0, 0, 1)
        awt = []
        for v in self.player.awaiting[city]:
            for item in self.player.awaiting[city][v]:
                awt.append(item)
        data['text'] = '' if len(awt)==0 else str(len(awt))
        awt = Counter(awt)
        data['hover'] = (self.capitalDisplay, city[0].upper()+city[1:]+' Awaiting Items | '+('None' if city=='fodker' else ', '.join([f'{a[0]}: [color=009e33]{a[1]}[/color]' for a in list(awt.items())])))
        return data
    def get_Capital(self):
        data = []
        for city in self.player.cityorder:
            data.append([self.skirmDisplay(city),
                         self.homeDisplay(city),
                         self.marketDisplay(city),
                         {'text':str(self.player.bank[city]) if self.player.bank[city] > 0 else '','background_color':(1,1,1,0),'color':(0,0,0,1), 'disabled':True},
                         self.villageDisplay(city),
                         self.villageAwaitingDisplay(city)])
        return data
    def get_holder(self, T):
        if T['maxRecord']['holder'] is None:
            holder = '-'
        elif T['maxRecord']['holder'] == self.player.username:
            holder = 'You'
        elif T['maxRecord']['value'] == T['value']:
            holder = 'Tied - but not held'
        else:
            holder = str(T['maxRecord']['value']) if T['maxRecord']['title'] != 'decisive' else toReadableTime(T['maxRecord']['value'], 2)
        return holder
    def get_Titles(self):
        data = []
        def as_button(text, title):
            return {'text': str(text), 'disabled': True, 'color': (0,0,0,1), 'hover': (self.titleDisplay, self.player.titles[title]['description'])}
        for i, title in enumerate(self.player.titleOrder):
            T = self.player.titles[title]
            data.append([as_button(str(i+1)+'. The '+title.capitalize(), title),
                         as_button(T['category'], title),
                         as_button(T['titleVP'], title),
                         as_button('-' if title=='decisive' else T['minTitleReq'], title),
                         as_button(T['value'] if title != 'decisive' else toReadableTime(T['value'], 2), title),
                         as_button(self.get_holder(T), title)])
        return data
    def update_single_title(self, title):
        i = self.player.titleIndex[title]
        self.Titles.cells['Your Record'][i].text = str(self.player.titles[title]['value']) if title != 'decisive' else toReadableTime(self.player.titles[title]['value'], 2)
        self.Titles.cells['Highest Record'][i].text = self.get_holder(self.player.titles[title])
        self.titlesTab.text = self.get_tab_text('Titles')
    def get_Items(self):
        data = []
        wares = self.player.currenttile.city_wares.union(self.player.currenttile.trader_wares)
        for item in self.player.items:
            categ, price = getItemInfo(item)
            if categ == 'Cloth':
                if self.player.currenttile.tile in item:
                    sellprice = 0
                else:
                    sellprice = price*(2 if self.player.currenttile.tile in clothSpecials[item] else 1) + self.player.bartering_mode
            elif price is None:
                sellprice = None
            else:
                sellprice = sellPrice[price] + self.player.bartering_mode
            clr = 'ffff75' if self.player.bartering_mode == 0 else '00ff75'
            cityset = self.player.currenttile.neighbortiles.intersection(set(cities))
            # Sell/Invest button:
            if sellprice is None:
                sellbutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            elif ((self.player.currenttile.trader_rounds>0) or (self.player.currenttile.tile in cities)) and (item not in wares):
                # Sell if the player is on a (city or trader) and the item is not in the wares
                sellbutton = {'text':f'Sell: [color={clr}]{sellprice}[/color]', 'markup':True, 'func':partial(sellItem, item, True if self.player.currenttile.trader_rounds>0 else False, None)}
            elif len(cityset) > 0:
                city = cityset.pop()
                if (self.player.currenttile.tile in {'village1','village2','village4','village5'}) and (categ == village_invest[self.player.currenttile.tile]):
                    # Invest if a player is on a village and the item is investable in the village
                    sellbutton = {'text':f'Invest: [color=ffff75]-{capital_info[city]["invest"]}[/color]', 'markup':True, 'func':partial(self.player.invest, item)}
                else:
                    # Empty
                    sellbutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            else:
                sellbutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            # Use button:
            if categ == 'Food':
                if 'raw' in item:
                    def eat_or_heat(_=None):
                        if self.player.paused:
                            return
                        output("Choose to Eat or Heat:", 'blue')
                        actionGrid({'Eat': partial(self.player.eat, item), 'Heat': partial(heatitem, item)}, False, False)
                    usebutton = {'text':'Eat/Heat', 'func':eat_or_heat}
                else:
                    usebutton = {'text':'Eat', 'func':partial(self.player.eat, item)}
            elif categ == 'Knowledge Books':
                usebutton = {'text':'Read', 'func':partial(readbook, item)}
            elif (item in heatableItems) and (item in noncraftableItems):
                usebutton = {'text':'Heat', 'func':partial(heatitem, item)}
            elif item == 'bark':
                def craft_or_heat(_=None):
                    if self.player.paused:
                        return
                    output("Choose to Craft or Heat:", 'blue')
                    actionGrid({'Heat':partial(heatitem, item), 'Craft':partial(self.player.PlayerTrack.craftingTable.add_craft, item, None)}, False, False)
                usebutton = {'text':'Craft/Heat', 'func':craft_or_heat}
            elif (categ == 'Crafting') and (item not in noncraftableItems):
                usebutton = {'text':'Craft', 'func':partial(self.player.PlayerTrack.craftingTable.add_craft, item, None)}
            elif (categ == 'Smithing') and (self.player.currenttile.tile in cities):
                usebutton = {'text':'Smith', 'func':partial(self.player.PlayerTrack.armoryTable.add_slot, item, None, None, None)}
            elif (categ == 'GrandLibrary') and (self.player.currenttile.tile in cities):
                book_split = ' '.split(item)
                level, skill = book_split[0].title(), (' '.join(book_split[1:])).title()
                if (skill in var.city_info[self.player.currenttile.tile]) and (var.city_info[self.player.currenttile.tile][skill]>=8) and self.player.training_allowed[self.player.currenttile.tile]:
                    usebutton = {'text': 'Learn', 'func': partial(learn_book, level, skill)}
                else:
                    usebutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            else:
                usebutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            data.append([{'text':item, 'background_color':(1,1,1,0), 'disabled':True,'color':(0,0,0,1)},
                         {'text':str(self.player.items[item]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         sellbutton,
                         usebutton])
            # [INCOMPLETE] Add an "invest" button option if player on a village
        return data
    def get_tab_text(self, category):
        category = category.capitalize()
        return f"{category}: [color={self.hclr}]{getattr(self.player, category)}[/color]"
    def updateTitles(self):
        self.titlesTab.text = self.get_tab_text('Titles')
        self.Titles.update_data_cells(self.get_Titles())
    def updateAll(self):
        self.Combat.update_data_cells(self.get_Combat(), False)
        self.Combat.update_header('Total', f'Total: {self.get_total_combat()}')
        self.combatTab.text=f"Combat: [color={self.hclr}]{self.player.Combat}[/color]"
        self.Knowledge.update_data_cells(self.get_Knowledge(), False)
        self.knowledgeTab.text=f'Knowledge: [color={self.hclr}]{self.player.Knowledge}[/color]'
        self.Capital.update_data_cells(self.get_Capital())
        self.capitalTab.text=f"Capital: [color={self.hclr}]{self.player.Capital}[/color]"
        self.fellowshipTab.text=self.get_tab_text('Fellowship')
        #self.updateTitles() # This should update on its own, no need to turn on this line.
        self.Items.update_data_cells(self.get_Items())
        self.itemsTab.text=f'Items: {self.player.item_count}/{self.player.max_capacity}'

#%% Player to Player Interactions: Trading

class TradePage(FloatLayout):
    def __init__(self, username, **kwargs):
        super().__init__(**kwargs)
        self.P = lclPlayer()
        self.P.paused = True
        self.P.is_trading = username
        self.O = username
        self.training = False
        self.is_trainee = False
        self.pconfirmed = False
        self.oconfirmed = False
        self.items = deepcopy(self.P.items)
        self.adept, self.master = {}, {}
        for skill, lvl in self.P.skills.items():
            if lvl >= 12:
                self.master.add(skill)
            elif lvl >= 8:
                self.adept.add(skill)
        for atr in self.P.attributes:
            lvl = self.P.combat[self.P.attributes[atr]]
            if lvl >= 12:
                self.master.add(atr)
            elif lvl >= 8:
                self.adept.add(atr)

        bkgSource = f'images\\resized\\background\\{self.P.currenttile.tile}.png' if (self.P.currenttile.tile+'.png') in os.listdir('images\\background') else f'images\\resized\\background\\{self.P.currenttile.tile[:-1]}.png'
        self.add_widget(Image(source=bkgSource, pos=(0,0), size_hint=(1,1)))

        psource = f'images\\resized\\origsize\\{self.P.birthcity}.png'
        x_person, y_person = 0.125, 0.3
        x_grid, y_grid = 0.3, 0.7
        x_middle, y_middle = 0.15, 0.7
        y_bottom = 0.3

        # Plot Player
        self.pimg = Image(source=psource, pos_hint={'x':0, 'y':y_bottom}, size_hint=(x_person, y_person))
        self.add_widget(self.pimg)
        # Plot Other Player
        osource = f'images\\resized\\origsize\\{self.P.parentBoard.Players[self.O].birthcity}.png'
        self.oimg = Image(source=osource, pos_hint={'right':1, 'y':y_bottom}, size_hint=(x_person, y_person))
        self.add_widget(self.oimg)

        pGrid = GridLayout(cols=3, pos_hint={'x':x_person, 'y':y_bottom}, size_hint=(x_grid, y_grid))
        mGrid = GridLayout(cols=1, pos_hint={'x':(x_person+x_grid), 'y':y_bottom}, size_hint=(x_middle, y_middle))
        oGrid = GridLayout(cols=3, pos_hint={'right':(1-x_person),'y':y_bottom}, size_hint=(x_grid, y_grid))

        slot_shape = (7, 3)
        self.offered = [{}, {}]
        self.slots = [np.empty(slot_shape,dtype=object), np.empty(slot_shape,dtype=object)]
        for i in range(slot_shape[0]):
            for j in range(slot_shape[1]):
                for side in [0, 1]:
                    self.slots[side][i,j] = Button(text='', disabled=True)
                    self.slots[side][i,j].amt = 0
                    self.slots[side][i,j].offered = None
                    if side == 0:
                        self.slots[side][i, j].bind(on_press=partial(self.remove, i, j, 0))
                        pGrid.add_widget(self.slots[side][i, j])
                    else:
                        oGrid.add_widget(self.slots[side][i, j])

        self.confirm = Button(text='Accept', color=(0.3, 1, 0.3, 1), background_color=(1.3, 1.5, 1.3, 1))
        self.confirm.bind(on_press=self.make_confirmation)
        self.decline = Button(text='Decline', color=(1, 0.3, 0.3, 1), background_color=(1.5, 1.3, 1.3, 1))
        self.decline.bind(on_press=self.reject_proposal)
        self.abort = Button(text='Abort', color=(0.8, 0, 0, 1), background_color=(1.7, 1.4, 1.4, 1))
        self.abort.bind(on_press=partial(self.quit_trade, 0, True))
        self.display = Button(text='', color=(0, 0, 0, 1), background_color=(1, 1, 1, 1), disabled=True, background_disabled_normal='')
        self.display.text_size = self.display.size
        self.display.bind(size=self.update_bkgSize)
        mGrid.add_widget(Widget())
        mGrid.add_widget(self.confirm)
        mGrid.add_widget(Widget())
        mGrid.add_widget(self.decline)
        mGrid.add_widget(Widget())
        mGrid.add_widget(Widget())
        mGrid.add_widget(self.abort)
        mGrid.add_widget(self.display)

        buffer, max_col = 0.95, 11
        coinGrid = GridLayout(cols=max_col, pos_hint={'x':0, 'top':y_bottom}, size_hint=(1, buffer*y_bottom/3))
        coinGrid.add_widget(Button(text="Add Coins:", disabled=True, background_disabled_normal='', color=(0,0,0,1)))
        for add in [1, 5, 25]:
            B = Button(text=f'+{add}', font_size=16, background_color=(1, 1, 0, 1))
            B.bind(on_press=partial(self.add_coins, add, 0, None))
            coinGrid.add_widget(B)

        itemGrid = GridLayout(cols=max_col, pos_hint={'x':0, 'top':y_bottom-(y_bottom/3)}, size_hint=(1, buffer*y_bottom/3))
        itemGrid.add_widget(Button(text='Add Item:', disabled=True, background_disabled_normal='', color=(0,0,0,1)))
        self.item_buttons = {}
        for item in self.items:
            B = Button(text=f'{item}: {self.items[item]}')
            B.bind(on_press=partial(self.add_item, item, 0, None))
            self.item_buttons[item] = B
            itemGrid.add_widget(B)

        trainGrid = GridLayout(cols=max_col, pos_hint={'x':0, 'top':y_bottom-2*(y_bottom/3)}, size_hint=(1, buffer*y_bottom/3))
        trainGrid.add_widget(Button(text='Offer Training:', disabled=True, background_disabled_normal='', color=(0,0,0,1)))
        for ability in self.adept:
            B = Button(text=f'Adept\n{ability}', background_color=(0.5, 0, 1, 1))
            B.bind(on_press=partial(self.add_train, ability, 0, 0, None))
            trainGrid.add_widget(B)
        for ability in self.master:
            B = Button(text=f'Master\n{ability}', background_color=(1, 1.5, 0, 1))
            B.bind(on_press=partial(self.add_train, ability, 1, 0, None))
            trainGrid.add_widget(B)

        self.add_widget(pGrid)
        self.add_widget(mGrid)
        self.add_widget(oGrid)
        self.add_widget(coinGrid)
        self.add_widget(itemGrid)
        self.add_widget(trainGrid)
    def update_bkgSize(self, instance, value):
        self.display.text_size = self.display.size
    def training_failure(self, msg='Training was unsuccessful', clr='red', _=None):
        output(msg, clr)
        self.refresh("Training was unsuccessful.", 3, (0.7, 0, 0, 1))
        self.P.takeAction()
    def receive_training(self, _=None):
        if self.P.actions <= 0:
            # They must have trained already and therefore they must decline
            self.reject_proposal()
            output("You have already ended the round. Cannot conduct training!", 'yellow')
            return
        ability = self.is_trainee[0]
        lvl = self.P.get_level(ability)
        requirement = 'go to combat' if ability in self.P.attributes else 'use the skill successfully'
        if lvl >= 12:
            output(f"You have already maxed out {self.is_trainee[0]}!", 'yellow')
            self.reject_proposal()
        if not self.is_trainee[1]:
            # Being taught by adept trainer.
            if lvl >= 8:
                output("Your level is too high to be taught by this trainer!", 'yellow')
                self.reject_proposal()
            elif rbtwn(1,10) <= self.P.fatigue:
                self.training_failure("You were unable to keep up with training.",'red')
                socket_client.send('[TRADE]', [self.O, 'training failure'])
            elif rbtwn(1,3) == 1:
                output(f"You successfully leveled up in {ability}",'green')
                newlevel = self.P.levelup(ability,1,8)
                self.P.updateTitleValue('apprentice', newlevel)
                socket_client.send('[TRADE]', [self.O, 'training success'])
                self.complete_trade()
            else:
                critthink = self.P.activateSkill('Critical Thinking')
                if rbtwn(1,12) <= critthink:
                    self.P.useSkill('Critical Thinking')
                    output(f"You successfully leveled up in {ability}",'green')
                    newlevel = self.P.levelup(ability,1,8)
                    self.P.updateTitleValue('apprentice', newlevel)
                    socket_client.send('[TRADE]', [self.O, 'training success'])
                    self.complete_trade()
                else:
                    self.training_failure()
                    socket_client.send('[TRADE]', [self.O, 'training failure'])
        else:
            # Being taught by master trainer.
            if rbtwn(2, 10) <= self.P.fatigue:
                self.training_failure("You were unable to keep up with training.",'red')
                socket_client.send('[TRADE]', [self.O, 'training failure'])
            elif lvl < 8:
                output(f"You successfully leveled up in {ability}",'green')
                newlevel = self.P.levelup(ability,1)
                self.P.updateTitleValue('apprentice', newlevel)
                socket_client.send('[TRADE]', [self.O, 'training success'])
                self.complete_trade()
            elif self.P.trained_abilities[ability]:
                output(f"You have already unlocked the potential to level up {ability}, now {requirement} to level up! Aborting trade.", 'red')
                socket_client.send('[TRADE]', [self.O, 'training failure'])
            elif rbtwn(1,4) == 1:
                output(f"You successfully unlocked the potential to level up {ability}! Now {requirement} to level up!", 'green')
                #self.P.levelup(ability,1)
                self.P.trained_abilities[ability] = True
                socket_client.send('[TRADE]', [self.O, 'training success'])
                self.complete_trade()
            else:
                critthink = self.P.activateSkill('Critical Thinking')
                if rbtwn(1,16) <= critthink:
                    self.P.useSkill('Critical Thinking', 2, 7)
                    output(f"You successfully unlocked the potential to level up {ability}! Now {requirement} to level up!", 'green')
                    #self.P.levelup(ability,1)
                    self.P.trained_abilities[ability] = True
                    socket_client.send('[TRADE]', [self.O, 'training success'])
                    self.complete_trade()
                else:
                    self.training_failure()
                    socket_client.send('[TRADE]', [self.O, 'training failure'])
    def complete_trade(self, _=None):
        consume = 'minor'
        for item, slot in self.offered[0].items():
            if type(item) is tuple:
                # Assumption is that training has been completed.
                consume = None
            elif item == 'coins':
                self.P.coins -= self.slots[0][slot].amt
            else:
                self.P.items.addItem(item, -self.slots[0][slot].amt)
        for item, slot in self.offered[1].items():
            if type(item) is tuple:
                consume = None
            elif item == 'coins':
                self.P.coins += self.slots[1][slot].amt
            else:
                self.P.items.addItem(item, self.slots[0][slot].amt)
        self.quit_trade(amt=consume)
    def make_confirmation(self, _=None):
        if self.P.actions > 0:
            # Offer can only be accepted if they have actions remaining.
            self.pconfirmed = True
            self.confirm.disabled = True
            if self.oconfirmed and self.training:
                if self.is_trainee:
                    self.receive_training()
            elif self.oconfirmed:
                self.complete_trade()
            else:
                self.display_msg(f"Awaiting {self.O} to accept offer.", None)
            socket_client.send('[TRADE]', [self.O, 'confirmed'])
        else:
            self.display_msg("You have no actions left! Either wait for the round to start or quit trade!", 10, (0.7, 0, 0, 1))
            socket_client.send('[TRADE]', [self.O, 'no actions'])
    def other_confirmation(self, _=None):
        self.oconfirmed = True
        if self.pconfirmed and self.training:
            if self.is_trainee:
                self.receive_training()
        elif self.pconfirmed:
            self.complete_trade()
        else:
            self.display_msg(f"{self.O} has accepted the offer!", None, (0, 0.7, 0, 1))
    def refresh(self, message='An update has been made.', delay=10, color=(0,0,0,1), _=None):
        self.pconfirmed = False
        self.confirm.disabled = False
        self.oconfirmed = False
        self.display_msg('')
        if message is not None:
            self.display_msg(message, delay, color)
    def reject_proposal(self, _=None):
        self.refresh('You rejected the offer!', 10, (0.7, 0, 0, 1))
        socket_client.send('[TRADE]', [self.O, 'decline'])
    def receive_rejection(self, _=None):
        self.refresh(f'{self.O} rejected the offer!', 10, (0.7, 0, 0, 1))
    def quit_trade(self, amt=0, send=False, _=None):
        if send:
            output(f"Trading with {self.O} has been aborted.")
            socket_client.send('[TRADE]', [self.O, 'abort'])
        self.P.trading = False
        self.P.parentBoard.game_page.main_screen.current = "Board" if self.P.parentBoard.game_page.toggleView.text=="Player Track" else "Player Track"
        self.P.parentBoard.game_page.main_screen.remove_widget(self.P.parentBoard.game_page.tradescreen)
        self.P.paused = False
        exitActionLoop(amt=amt)()
    def display_msg(self, msg, delay=None, color=(0, 0, 0, 1)):
        previous_message, previous_color = self.display.text, self.display.color
        if delay is not None:
            self.display.override = True
        else:
            self.display.override = False
        def clear(_=None):
            if self.display.override:
                self.display.text = previous_message
                self.display.color = previous_color
        self.display.text = msg
        self.display.color = color
        if delay is not None: Clocked(clear, delay, 'clear display message')
    def remove(self, i, j, side=0, _=None):
        if type(self.slots[side][i, j].offered) is tuple:
            self.training = False
            self.confirm.text = 'Accept'
        if side == 0:
            socket_client.send('[TRADE]', [self.O, self.slots[side][i, j].offered, -1, (i, j)])
            if (type(self.slots[side][i, j].offered) is str) and not (self.slots[side][i, j].offered=='coins'):
                # Assume it must be an item
                item = self.slots[side][i, j].offered
                self.items[item] += self.slots[side][i, j].amt
                self.item_buttons[item].text = f'{item}: {self.items[item]}'
                self.item_buttons[item].disabled = False
        self.refresh()
        self.slots[side][i, j].amt = 0
        self.slots[side][i, j].text = ''
        self.slots[side][i, j].disabled = True
        self.offered[side].pop(self.slots[side][i, j].offered)
        self.slots[side][i, j].offered = None
    def findNearestEmpty(self, side):
        for i in range(len(self.slots[side])):
            for j in range(len(self.slots[side][i])):
                if self.slots[side][i, j].text == '':
                    return (i, j)
        self.display_msg("You cannot make anymore offers to the trade! Try removing some!", 10, (0.7, 0.7, 0, 1))
        return False
    def add_coins(self, amt, side=0, slot=None, _=None):
        if slot is None:
            if 'coins' in self.offered[side]:
                slot = self.offered[side]['coins']
            else:
                slot = self.findNearestEmpty(side)
                if not slot: return
        if not ((side == 0) and (self.P.coins < (amt + self.slots[side][slot].amt))):
            self.refresh()
            self.slots[side][slot].amt += amt
            self.slots[side][slot].text = f"{self.slots[side][slot].amt} Coin{'' if self.slots[side][slot].amt==1 else 's'}"
            self.slots[side][slot].offered = 'coins'
            self.offered[side]['coins'] = slot
            if side == 0:
                self.slots[side][slot].disabled = False
                socket_client.send('[TRADE]', [self.O, 'coins', amt, slot])
        else:
            self.display_msg("You do not have enough coin to make that offer!", 10, (0.7, 0.7, 0, 1))
    def add_item(self, item, side=0, slot=None, _=None):
        if slot is None:
            if item in self.offered[side]:
                slot = self.offered[side][item]
            else:
                slot = self.findNearestEmpty(side)
                if not slot: return
        if not ((side == 0) and (self.items[item] <= 0)):
            self.refresh()
            self.slots[side][slot].amt += 1
            self.slots[side][slot].text = f"{self.slots[side][slot].amt} {item}{'' if self.slots[side][slot].amt==1 else 's'}"
            self.slots[side][slot].offered = item
            self.offered[side][item] = slot
            if side == 0:
                self.items[item] -= 1
                self.item_buttons[item].text = f'{item}: {self.items[item]}'
                self.slots[side][slot].disabled = False
                if self.items[item] <= 0:
                    self.item_buttons[item].disabled = True
                socket_client.send('[TRADE]', [self.O, item, 1, slot])
        else:
            self.display_msg("You cannot offer any more of that item!", 10, (0.7, 0.7, 0, 1))
    def add_train(self, ability, master, side=0, slot=None, _=None):
        if self.training:
            self.display_msg("You cannot train more than one at a time!", 10, (0.7, 0.7, 0, 1))
            return
        if slot is None:
            slot = self.findNearestEmpty(side)
            if not slot: return
        self.refresh()
        self.training = True
        self.confirm.text = 'Train'
        self.slots[side][slot].amt = master
        self.slots[side][slot].text = "{'Master' if master else 'Adept'} {ability}\nTraining"
        self.slots[side][slot].offered = (ability, master)
        self.offered[side][(ability, master)] = slot
        if side == 0:
            self.slots[side][slot].disabled = False
            socket_client.send('[TRADE]', [self.O, (ability, master), None, slot])
        else:
            self.is_trainee = (ability, master)

def player_trade(username, _=None):
    P = lclPlayer()
    screen = Screen(name="Trade")
    screen.add_widget(TradePage(username))
    P.parentBoard.game_page.main_screen.add_widget(screen)
    P.parentBoard.game_page.main_screen.current = "Trade"
    P.parentBoard.game_page.tradescreen = screen

def player_confirm_trade(username, _=None):
    P = lclPlayer()
    def reject_trade(_=None):
        output(f"{username} attempted to trade with you but was rejected.", 'yellow')
        socket_client.send('[TRADE]', {username: 'reject'})
        exitActionLoop(amt=0)()
    def start_trade(_=None):
        if P.parentBoard.Players[username].currenttile == P.currenttile:
            # Make sure they are still on the same tile
            socket_client.send('[TRADE]', {username: 'start'})
            player_trade(username)
        else:
            output(f"You and {username} are no longer on the same tile. Cannot trade.", 'yellow')
            reject_trade()
    if P.paused or P.parentBoard.game_page.occupied:
        reject_trade()
    else:
        output(f"{username} wants to trade with you. Conduct trade?", 'blue')
        actionGrid({'Yes': start_trade, 'No': reject_trade}, False)

def player_ask_trade(username, _=None):
    P = lclPlayer()
    if P.paused:
        return
    output(f"Sending trade request to {username}.", 'blue')
    socket_client.send('[TRADE]', {username: 'ask'})

#%% City and Board Properties
cities = var.cities
city_villages = var.city_villages
sellPrice = var.sellPrice
mrktPrice = var.mrktPrice
price2smithlvl = var.price2smithlvl
gameItems = var.gameItems
clothSpecials = var.clothSpecials
valid_items = var.valid_items

game_launched = [False]
hovering = [0]

def getItemInfo(item):
    for categ in gameItems:
        if item in gameItems[categ]:
            return categ, gameItems[categ][item]
    output(f"Unrecognized item {item}!",'yellow')

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

positions = var.positions
randoms = var.randoms

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

def city_trainer(abilities, mastery, _=None):
    actions = {ability:partial(Train, ability, mastery, False) for ability in abilities}
    actions["Back"] = exitActionLoop(amt=0)
    actionGrid(actions, False)

skill_users = {'Persuasion':'Politician', 'Critical Thinking':'Librarian', 'Heating':'Chef', 'Survival':'Explorer', 'Smithing':'Smith', 'Crafting':'Innovator', 'Excavating':'Miner', 'Stealth':'General', 'Gathering':'Huntsman', 'Bartering':'Merchant'}

def perform_labor(skill=None, _=None):
    P = lclPlayer()
    if skill is not None:
        P.working = [skill, 1]
    else:
        P.working[1] += 1
    if P.working[1] >= 4:
        skill = P.working[0]
        P.coins += P.skills[skill]
        P.addXP(skill, 0 if P.skills[skill] > 7 else max([1, P.skills[skill]//2]))
        P.working = [None, 0]
        P.takeAction(1)
        # Update title values
        P.updateTitleValue('laborer', 1)
        P.updateTitleValue('valuable', P.skills[skill])
        # Start next round
        P.go2action()
    else:
        P.takeAction(1)

def confirm_labor(skill, _=None):
    P = lclPlayer()
    if P.paused:
        return
    payment = P.skills[skill]
    xp = 0 if P.skills[skill] > 7 else max([1, P.skills[skill]//2])
    if (P.fatigue + 5) >= P.max_fatigue:
        output(f"You could have received {payment} coins and {xp} {skill} XP in 4 actions, if your fatigue wasn't so high.", 'yellow')
    else:
        if payment == 0:
            output("Your level is zero! You will not be paid for this job! However, you can still volunteer.", 'yellow')
        output(f'Would you like to assist {skill_users[skill]} for 4 actions? [Payment={payment}, {skill} XP={xp}, 4 fatigue applied afterwards]', 'blue')
        actionGrid({'Yes':partial(perform_labor, skill), 'No':exitActionLoop(amt=0)}, False, False)

def city_labor(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if (P.fatigue + 5) >= P.max_fatigue:
        output(f"Uh oh, your fatigue must be less than {P.max_fatigue - 4} to go to work! You can still click the jobs for info.", 'yellow')
    else:
        output("The following people are looking for assistants today:", 'blue')
    actions = {'Cancel':exitActionLoop(amt=0)}
    for skill in P.currenttile.city_jobs:
        actions[skill_users[skill]] = partial(confirm_labor, skill)
    actionGrid(actions, False)

def TendMarket(_=None):
    P = lclPlayer()
    if P.paused:
        return
    if P.tended_market[P.currenttile.tile]:
        output("You already tended the market this round, you will need to wait until next round!", 'yellow')
        exitActionLoop(amt=0)()
        return
    notwaring = notAtWar(Skirmishes[0])
    if P.currenttile.tile not in notwaring:
        output("The city is currently in a skirmish - so there is little activity in the market. Try again later.", 'yellow')
        exitActionLoop(amt=0)()
        return
    P.tended_market[P.currenttile.tile] = True
    output("You tended the market this round. Stay in the city until the end of the round to receive your income.", 'green')
    exitActionLoop(amt=1)()

def city_consequence(city):
    P = lclPlayer()
    # Check if tensions exist between the cities
    if city in Skirmishes[2][P.birthcity]:
        # Check if currently in skirmish
        S = frozenset([city, P.birthcity])
        if S in Skirmishes[0]:
            # 1/3 probability that hooligan is aggrevated
            if rbtwn(0,2) == 0:
                output("Hooligans are aggrevated!", 'yellow')
                # Check if stealth saves you
                stealth = P.activateSkill("Stealth")
                r = rbtwn(0, 12, None, stealth, 'Stealth ')
                if r <= stealth:
                    P.useSkill("Stealth")
                    output("You are able to avoid hooligan's detection!", 'green')
                else:
                    persuasion = P.activateSkill("Persuasion")
                    r = rbtwn(0, 12, None, persuasion, "Persuasion ")
                    if r <= persuasion:
                        P.useSkill("Persuasion")
                        output("You are able to convince the hooligan you mean no harm.", 'green')
                    else:
                        encounter(f'{city[0].upper()+city[1:]} Hooligan', [15, 80], [cities[city]['Combat Style']], {'coins':[1, 8]})
                        return
            else:
                output(f"Be mindful: Tensions are active between {city} and {P.birthcity}! Hooligans could be aggrevated!", 'yellow')
        else:
            output(f"Tensions between {city} and {P.birthcity} are calm at the moment.")
    exitActionLoop()()

def city_actions(city, _=None):
    P = lclPlayer()
    def Sparring(_=None):
        if not P.paused:
            encounter('Sparring Partner', [2, 5], [cities[city]['Combat Style']], {}, enc=0)
    def Duelling(_=None):
        if (not P.paused) and (P.dueling_hiatus==0):
            P.dueling_hiatus = 2
            encounter('Duelist', [max([2, sum(P.current)-2]), sum(P.current)+6], ['Physical', 'Elemental', 'Trooper', 'Wizard'], {'coins':int(P.Combat/10)+2}, enc=0)
        elif P.paused and (P.dueling_hiatus>0):
            output(f"You still have to wait {P.dueling_hiatus} more rounds to duel!", 'yellow')
    def Inn(_=None):
        if P.paused:
            return
        if P.coins == 0:
            output("Insufficient coin", 'yellow')
            return
        P.coins -= 1
        P.recover(3)
    logging.debug(f"Generating city actions for {city}.")
    T = game_app.game_page.board_page.citytiles[city]
    actions = {}
    actions['Market'] = partial(Trading, False)
    actions['Job Postings'] = city_labor
    if P.training_allowed[P.currenttile.tile]:
        actions['Adept'] = partial(city_trainer, T.adept_trainers, 'adept')
        actions['Master'] = partial(city_trainer, T.master_trainers, 'city')
    if P.markets[P.currenttile.tile]:
        actions['Tend Market'] = TendMarket
    if (P is not None) and (P.currenttile.tile == 'scetcher'):
        actions['Duel'] = Duelling
    elif (P is not None) and (P.birthcity == city):
        actions['Sparring'] = Sparring
    if (P is not None) and (P.birthcity != city) and (not P.homes[city]):
        actions['Inn (3): 1 coin'] = Inn
    if P.currenttile.tile == P.birthcity:
        # If player is in their birth city then allow more actions depending on active quests
        actions = P.PlayerTrack.Quest.add_active_city_actions(actions)
    # Add any special quest actions:
    if (P.PlayerTrack.Quest.quests[2, 8].status=='started') and (P.currenttile.tile in {'enfeir', 'zinzibar'}):
        actions['Approach Stealth Master'] = PursuadeStealthMaster
    if (P.PlayerTrack.Quest.quests[4, 2].status=='started') and (P.currenttile.tile in cities) and (P.currenttile.tile != P.birthcity):
        if not (hasattr(P.PlayerTrack.Quest.quests[4, 2], 'cities_searched') and (P.currenttile.tile in P.PlayerTrack.Quest.quests[4, 2].cities_searched)):
            actions["Find Warrior"] = FindWarrior
    if (P.PlayerTrack.Quest.quests[4, 5].status=='started') and ({P.currenttile.tile, P.birthcity} in Skirmishes[1]):
        actions["Find & Convince Leader"] = FindAndPursuadeLeader
    if (P.PlayerTrack.Quest.quests[4, 8].status=='started') and (P.currenttile.tile not in Skirmishes[2][P.birthcity]):
        actions["Find & Convince Warrior"] = ConvinceWarriorsToRaid
    elif (P.PlayerTrack.Quest.quests[4, 8].status=='completed') and (P.currenttile.tile in P.PlayerTrack.Quest.quests[4, 8].cities):
        actions["Add Warrior to Group"] = ConvinceWarriorToJoin
    if (P.PlayerTrack.Quest.quests[5, 5].status == 'started') and (P.currenttile.tile == 'demetry'):
        actions["Convince Bartering Master"] = ConvinceBarterMaster
    if (P.PlayerTrack.Quest.quests[5, 7].status=='started') and (P.currentcoord == P.PlayerTrack.Quest.quests[2, 8].coord):
        actions["Approach Stealth Master"] = AskMasterStealthForMayor
    actionGrid(actions, True)

class CityPage(ButtonBehavior, HoverBehavior, FloatLayout):
    def __init__(self, city, player=None, **kwargs):
        super(CityPage, self).__init__(**kwargs)
        logging.debug(f"Initiating City Page for {city}")
        self.city = city
        self.P = game_app.game_page.board_page.localPlayer if player is None else player

        # Set City Values
        image_source = f'images\\cities\\{city}.png'
        self.img = Image(source=image_source, pos=(0,0), size_hint=(1,1))
        self.img.allow_stretch = True
        self.img.fit_mode = 'contain'
        self.add_widget(self.img)

        self.pil_img = PILImage.open(f'images/cities/region/{city}.png') # eventually this will be replaced with city-specific regions
        self.slct_img = None
        self.slct_region = None
        self.hover_label = None
        self.filter_color = Color(0, 0, 0, 0)
        self.filter_rect = None
        self.draw_filter() # almost makes the two lines above redundant.
        self.persons_active = {}
        self.quest_buttons_active = []
        self.persons_active_found = False

        Window.bind(mouse_pos=self.on_mouse_pos)
        self.bind(on_press=self.activate)

    def draw_filter(self):
        with self.img.canvas.after:
            self.filter_color = Color(0, 0, 0, .1)  # Semi-transparent black filter
            self.filter_rect = Rectangle(pos=self.img.pos, size=self.img.size)
            # Initially, make the filter fully transparent
            self.filter_color.rgba = (0, 0, 0, 0)

    def update_filter_visibility(self, visible):
        if visible:
            self.filter_color.rgba = (0, 0, 0, .1)  # Make filter visible
        else:
            self.filter_color.rgba = (0, 0, 0, 0)  # Make filter invisible

    def get_actionGrid(self):
        actionGrid = game_app.game_page.get_actionGrid()
        actionGrid[game_app.game_page.recover_button.text] = lclPlayer().recover
        return actionGrid

    def switch2board(self, instance):
        game_app.game_page.main_screen.current = 'Board'

    def enable_quest(self, stage, mission):
        person = var.inverse_quest_mapper[(stage, mission)]
        if person in self.persons_active:
            self.persons_active[person]['quests'].add((stage, mission))
        else:
            self.persons_active[person] = {'quests': {(stage, mission)}, 'img': None}
            # add explanation point hover over person
            mnx, mxx, mny, mxy = pqp.person_pos[person]
            xoffset = float(((mnx - mxx) * 0.25) / pqp.x_len)
            pos_hint = {'x': float(mnx / pqp.x_len) + xoffset, 'y': float((mxy + 5) / pqp.y_len)}
            size_hint = (1.5 * float((mxx - mnx) / pqp.x_len), 1.5 * float((mxx - mnx) / pqp.y_len))
            logging.debug(f'{person}, {pos_hint}, {size_hint}')
            exp_img = Image(source='images/assets/quest_alert.png', pos_hint=pos_hint, size_hint=size_hint)
            self.persons_active[person]['img'] = exp_img
            self.add_widget(exp_img)

    def find_active_quests(self, force=False):
        if (not self.persons_active_found) or force:
            self.persons_active_found = True
            for stage, mission in var.inverse_quest_mapper:
                quest = getQuest(stage, mission)
                if not quest.disabled:
                    self.enable_quest(stage, mission)

    def disable_active_quest(self, stage, mission):
        person = var.inverse_quest_mapper[(stage, mission)]
        if (person in self.persons_active):
            try:
                self.persons_active[person]['quests'].remove((stage, mission))
            except KeyError:
                return
            if len(self.persons_active[person]['quests']) == 0:
                # at this point delete the explanation point hovering over person
                self.remove_widget(self.persons_active[person]['img'])
                self.persons_active.pop(person)

    def add_hover_label(self, region, pos):
        if pos is not None:
            txt = var.action_mapper[region] if region in var.action_mapper else ' '.join([l.capitalize() for l in region.split('_')])
            self.hover_label = HoveringLabel(text=txt, color=(1, 1, 1, 1), markup=True, pos=pos,
                                             size_hint=(0.0075 * len(txt), 0.03))
            # self.hover_label = Button(text=txt, color=(1,1,1,1), markup=True, pos=pos, size_hint=(None, None), background_color=(0.3, 0.3, 0.3, 0.7)) # deprecate b/c the button was getting in the way of clicking the image
            self.add_widget(self.hover_label)

    def remove_hover_label(self):
        if self.hover_label is not None:
            self.remove_widget(self.hover_label)
            self.hover_label = None
    def select(self, region, pos=None):
        if self.slct_img is not None:
            self.remove_widget(self.slct_img)
            self.slct_img = None
            self.slct_region = None
            self.remove_hover_label()
        self.slct_region = region
        if region is not None:
            image_source = f'images\\cities\\selection\\{region}.png'
            self.slct_img = Image(source=image_source, pos=(0, 0), size_hint=(1, 1))
            self.slct_img.allow_stretch = True
            self.slct_img.fit_mode = 'contain'
            self.add_widget(self.slct_img)
            self.add_hover_label(region, pos)

    def _gates(self):
        self.switch2board(None)
    def remove_multi_options(self):
        outside = all(not button.collide_point(*Window.mouse_pos) for button in self.quest_buttons_active)
        if outside:
            while len(self.quest_buttons_active) > 0:
                button = self.quest_buttons_active.pop()
                self.remove_widget(button)
            self.update_filter_visibility(False)
    def add_multi_options(self, texts, functions):
        pos = Window.mouse_pos
        size_hint = var.multi_button_size
        l = len(texts)
        for i, (text, func) in zip(texts, functions):
            b_pos = (pos[0] - (size_hint[0] * l) / 2 + (size_hint[0] * i), pos[1] + size_hint[1])
            H = Button(text=text, pos=b_pos, size_hint=size_hint)
            H.bind(on_press=func)
            self.quest_buttons_active.append(H)
            self.add_widget(H)
        self.update_filter_visibility(True)
    def activate(self, instance):
        logging.debug(f'clicking region {self.slct_region}')
        if game_app.game_page.main_screen.current != 'City':
            return False
        
        if lclPlayer().paused:
            output("You cannot perform any more actions this round.", 'yellow')
            return

        if len(self.quest_buttons_active) > 0:
            self.remove_multi_options()

        elif self.slct_region is not None:
            if hasattr(self, f'_{self.slct_region}'):
                getattr(self, f'_{self.slct_region}')()
            elif self.slct_region in var.action_mapper:
                actionGrid = self.get_actionGrid()
                action = var.action_mapper[self.slct_region]
                if action in actionGrid:
                    actionGrid[action]()
                elif self.slct_region == 'shack':
                    output('You no longer live in the shack! You can rest at home.', 'yellow')
                else:
                    output('This option is not available to you!', 'yellow')
            elif self.slct_region in var.region_quest_mapper:
                if len(var.region_quest_mapper[self.slct_region]) == 1:
                    stage, mission = var.region_quest_mapper[self.slct_region][0]
                    B = getQuest(stage, mission) # quest button
                    lclPlayer().PlayerTrack.Quest.activate(B)
                else:
                    texts, functions = [], []
                    for stage, mission in var.region_quest_mapper[self.slct_region]:
                        B = getQuest(stage, mission)
                        functions.append(partial(lclPlayer().PlayerTrack.Quest.activate, B))
                        texts.append(B.text)
                    self.add_multi_options(texts, functions)
            elif self.slct_region in var.hallmark_mapper:
                self.P.fellowships[var.hallmark_mapper[self.slct_region]].begin_action_loop()

    def on_mouse_pos(self, window, pos):
        # Ensure that the current main_page is the CityPage
        if game_app.game_page.main_screen.current != 'City':
            return False

        # Check if the mouse is over the widget - perhaps redundant to the above check.
        if not self.img.collide_point(*pos):
            return

        # if quest buttons are active then don't update these select widgets.
        if len(self.quest_buttons_active) > 0:
            return

        # Get the position and size of the widget
        wx, wy = self.img.pos
        w_width, w_height = self.img.size

        # Calculate the scale factor and offset based on 'contain' mode
        img_ratio = self.pil_img.width / self.pil_img.height
        widget_ratio = w_width / w_height
        if img_ratio > widget_ratio:
            # Image is wider than the widget
            scale = w_width / self.pil_img.width
            offset_x = 0
            offset_y = (w_height - (self.pil_img.height * scale)) / 2
        else:
            # Image is taller than the widget
            scale = w_height / self.pil_img.height
            offset_x = (w_width - (self.pil_img.width * scale)) / 2
            offset_y = 0

        # Adjust mouse coordinates to image coordinates
        mx, my = pos[0] - wx - offset_x, pos[1] - wy - offset_y
        nx, ny = int(mx / scale), int(my / scale)

        # Adjust for the y coordinate being inverted in PIL
        ny = self.pil_img.height - ny - 1

        if (0 <= nx < self.pil_img.width) and (0 <= ny < self.pil_img.height):
            # Get the color of the pixel
            pixel = self.pil_img.getpixel((nx, ny))
            hex_value = '%02x%02x%02x' % pixel[:3]
            region = var.region_colors[hex_value]

            unlock_region = False
            if region in var.region_quest_mapper:
                for stage, mission in var.region_quest_mapper[region]:
                    quest = getQuest(stage, mission)
                    if not quest.disabled:
                        unlock_region = True # if even one quest is unlocked, allow the highlight
            else:
                unlock_region = True

            if unlock_region and (region != self.slct_region):
                self.select(region, pos)

        self.find_active_quests()

class Tile(ButtonBehavior, HoverBehavior, Image):
    def __init__(self, tile, x, y, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.source = f'images\\tile\\{tile}.png'
        self.hoveringOver = False
        self.entered = False
        self.parentBoard = None
        self.empty_label = None
        self.empty_label_rounds = None
        self.is_empty = False
        self.trader_id = None
        self.trader_label = None
        self.trader_rounds = 0
        self.trader_wares = set()
        self.trader_wares_label = None
        self.city_wares = set()
        self.city_jobs = set()
        self.adept_trainers = set()
        self.master_trainers = set()
        self.neighbors = set()
        self.lockIcon = None
        self.skirmishIcon = None
        self.traveledOn = False
        self.bind(on_press=self.initiate)
        Window.bind(mouse_pos = self.on_mouse_move)
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
            self.parentBoard.traderID += 1
            self.trader_id = self.parentBoard.traderID
        else:
            self.trader_wares = recvd
        self.color = (0.5, 1, 0.5, 1)
        self.trader_label = Label(text=str(rounds), bold=True, pos_hint=self.pos_hint, size_hint=self.size_hint, color=(0,0,0,1), markup=True, font_size=20)
        self.trader_rounds = rounds
        self.parentBoard.add_widget(self.trader_label)
        self.parentBoard.sendFaceMessage('Trader Appears!','green')
    def remove_trader(self, recvd=False, _=None):
        if not recvd:
            socket_client.send('[TRADER]', [(self.gridx, self.gridy), 0])
        self.trader_rounds = 0
        self.trader_label.text = ''
        self.parentBoard.remove_widget(self.trader_label)
        self.trader_label = None
        self.trader_id = None
        self.color = (1, 1, 1, 1)
        self.trader_wares = set()
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
                self.remove_trader(True)
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
        if self.lockIcon is not None:
            self.lockIcon.size_hint = self.size_hint
            self.lockIcon.pos_hint = self.pos_hint
        if self.skirmishIcon is not None:
            self.skirmishIcon.size_hint = self.size_hint
            self.skirmishIcon.pos_hint = self.pos_hint
        self.centx, self.centy = xpos + xshift + (xprel*mag_x/2), ypos + yshift + (yprel*mag_y/2)
        # Get polygonal shape for mouse position detection
        self.update_polygon()
    def set_neighbors(self):
        neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1 if self.gridy % 2 else -1, 1], [1 if self.gridy % 2 else -1, -1]]
        self.neighbors, self.neighbortiles = set(), set()
        for dx, dy in neighbors:
            x, y = self.gridx + dx, self.gridy + dy
            if (x, y) in self.parentBoard.gridtiles:
                self.neighbors.add((x, y))
                self.neighbortiles.add(self.parentBoard.gridtiles[(x,y)].tile)
        if self.tile in cities:
            city_villages[self.tile] = sorted(self.neighbortiles.intersection({'village1','village2','village3','village4','village5'}))
    def is_neighbor(self):
        return self.parentBoard.localPlayer.currentcoord in self.neighbors
    def findNearest(self, tiles):
        T = self
        checked, queue = {(T.gridx, T.gridy):None}, [(self, 0)]
        while queue:
            T, depth = queue.pop(0)
            if T.tile in tiles:
                path, c = [], (T.gridx, T.gridy)
                while c is not None:
                    path.append(c)
                    c = checked[c]
                return (T.gridx, T.gridy), depth, path
            nextinLine = T.neighbors.difference(checked)
            for coord in nextinLine:
                checked[coord] = (T.gridx, T.gridy)
                queue.append((self.parentBoard.gridtiles[coord], depth+1))
        return None, None, None
    def on_mouse_move(self, *args):
        if self.entered:
            in_hexagon = self.check_mouse_in_hexagon()
            #logging.debug(f"Tile {self.tile} in hexagon: {in_hexagon}")
            if in_hexagon:
                self.hoveringOver = True
                #hovering[0] += 1
                if lclPlayer().currenttile != self:
                    self.source = f'images\\selectedtile\\{self.tile}.png'
                if (self.trader_rounds > 0) and (self.trader_wares_label is None):
                    L = []
                    for item in self.trader_wares:
                        categ, price = getItemInfo(item)
                        L.append(f'{item}: [color=ffff75]{gameItems[categ][item]}[/color]')
                    self.trader_wares_label = Button(text='\n'.join(L), color=(1, 1, 1, 1), markup=True, pos=(self.pos[0]+self.size[0]/2,self.pos[1]+self.size[1]/2), size_hint=(self.size_hint[0]*4/(self.parentBoard.zoom+1),self.size_hint[1]*1.5/(self.parentBoard.zoom+1)), background_color=(0.3, 0.3, 0.3, 0.7))
                    self.parentBoard.add_widget(self.trader_wares_label)
            elif self.hoveringOver:
                self.hoveringOver = False
                #hovering[0] -= 1
                if lclPlayer().currenttile == self:
                    self.source = f'images\\ontile\\{self.tile}.png'
                else:
                    self.source = f'images\\tile\\{self.tile}.png'
                if self.trader_wares_label is not None:
                    self.trader_wares_label.text = ''
                    self.parentBoard.remove_widget(self.trader_wares_label)
                    self.trader_wares_label = None
    def on_enter(self, *args):
        #logging.debug(f"Mouse is entering {self.tile}")
        self.entered = True
        self.update_polygon()
#        self.hoveringOver = True
#        hovering[0] += 1
#        if lclPlayer().currenttile != self:
#            self.source = f'images\\selectedtile\\{self.tile}.png'
#        if (self.trader_rounds > 0) and (self.trader_wares_label is None):
#            L = []
#            for item in self.trader_wares:
#                categ, price = getItemInfo(item)
#                L.append(f'{item}: [color=ffff75]{gameItems[categ][item]}[/color]')
#            self.trader_wares_label = Button(text='\n'.join(L), color=(1, 1, 1, 1), markup=True, pos=(self.pos[0]+self.size[0]/2,self.pos[1]+self.size[1]/2), size_hint=(self.size_hint[0]*4/(self.parentBoard.zoom+1),self.size_hint[1]*1.5/(self.parentBoard.zoom+1)), background_color=(0.3, 0.3, 0.3, 0.7))
#            self.parentBoard.add_widget(self.trader_wares_label)
    def on_leave(self, *args):
        #logging.debug(f"Mouse is leaving {self.tile}")
        self.entered = False
        self.hoveringOver = False
#        hovering[0] -= 1
        if lclPlayer().currenttile == self:
            self.source = f'images\\ontile\\{self.tile}.png'
        else:
            self.source = f'images\\tile\\{self.tile}.png'
        if self.trader_wares_label is not None:
            self.trader_wares_label.text = ''
            self.parentBoard.remove_widget(self.trader_wares_label)
            self.trader_wares_label = None
    def update_polygon(self):
        xp, yp = self.pos
        dx, dy = self.size
        ry = (1/np.sqrt(3)) * (dx/2) # Get height of triangle for hexagon y-pos
        self.polygon = mplPath.Path(np.array([[xp+dx/2, yp], [xp+dx, yp+ry], [xp+dx, yp+dy-ry], [xp+dx/2, yp+dy], [xp, yp+dy-ry], [xp, yp+ry]]))
    def check_mouse_in_hexagon(self):
        return self.polygon.contains_point(Window.mouse_pos)
    def initiate(self, instance):
        if self.parentBoard.inspect_mode:
            logging.debug(f"Tile: {self.tile}, Tile Position: {self.pos}, Center: {self.centx, self.centy}, Size: {self.size}")
            logging.debug(f"Mouse Position: {Window.mouse_pos}")
            logging.debug(f"Mouse in Hexagon: {self.check_mouse_in_hexagon()}")
            if self.tile in avail_actions:
                self.parentBoard.game_page.inspectTile(*avail_actions[self.tile](inspect=True))
            if self.tile in consequence_message:
                self.parentBoard.game_page.consequenceDisplay.text = consequence_message[self.tile]
            elif (self.tile in Skirmishes[2][self.parentBoard.localPlayer.birthcity]):
                self.parentBoard.game_page.consequenceDisplay.text = 'Hooligan encounter chance: 1/3. Hooligan Lvl: 15-80. Reward: 1-8 coins. Hooligan Avoidance: (stealth+1)/13, if fails then (persuasion+1)/13.'
            else:
                self.parentBoard.game_page.consequenceDisplay.text = ''
        elif self.parentBoard.localPlayer.teleport_ready:
            self.parentBoard.localPlayer.fellowships['tamariza'].teleport_to((self.gridx, self.gridy))
        else:
            if self.parentBoard.localPlayer.paused:
                return
            elif not self.check_mouse_in_hexagon():
                #output("Hovering over too many tiles! No action performed!",'yellow')
                return
            # If the tile is a neighbor and the player is not descended in a cave or ontop of a mountain... (tiered)
            elif self.is_neighbor() and (not self.parentBoard.localPlayer.tiered):
                self.parentBoard.localPlayer.moveto((self.gridx, self.gridy))
            # If they click the tile they are on, and they are on a city tile, then switch bck to the city page.
            elif (self.tile in cities) and (self == self.parentBoard.localPlayer.currenttile):
                self.parentBoard.game_page.main_screen.current = 'City'

trns_clr = {'red':essf.get_hexcolor((255,0,0)),'green':essf.get_hexcolor((0,255,0)),'yellow':essf.get_hexcolor((147,136,21)),'blue':essf.get_hexcolor((0,0,255)),'cyan':essf.get_hexcolor((0, 255, 255))}
class BoardPage(FloatLayout):
    def __init__(self, game_page, **kwargs):
        super().__init__(**kwargs)
        self.game_page = game_page
        self.localuser = game_app.connect_page.username.text
        self.size = get_dim(xtiles, ytiles)
        self.traderID = 0
        self.zoom = 0
        self.inspect_mode = False
        self.gridtiles = {}
        self.citytiles = {}
        self.Players = {}
        self.localPlayer = None
        self.startLblMsgs = []
        self.startLblTimes = []
        self.startLblLast = []
        self.defaultRoundDelay = 1
        for tiletype in positions:
            if tiletype == 'randoms':
                continue
            for x, y in positions[tiletype]: self.add_tile(tiletype, x, y)
        self.semiRandomTileDistribution()
        #self.completeRandomTileDistribution()
        for T in self.gridtiles.values():
            if T.tile in cities:
                self.citytiles[T.tile] = T
                # Initial markets / job postings will be different for local players
                T.city_wares = set(np.random.choice(list(city_info[T.tile]['sell']), 6))
                jobs, trns = set(), {8:5, 12:8, 4:2}
                for skill in skill_users:
                    val = trns[city_info[T.tile][skill]] if skill in city_info[T.tile] else 2
                    if rbtwn(1, 10) <= val:
                        jobs.add(skill)
                T.city_jobs = jobs
            T.set_neighbors()
        self.zoomButton = Button(text="Zoom", pos_hint={'x':0,'y':0}, size_hint=(0.06, 0.03))
        self.zoomButton.bind(on_press=self.updateView)
        self.add_widget(self.zoomButton)
        self.inspectButton = Button(text='Inspect', pos_hint={'x':0.07,'y':0}, size_hint=(0.06, 0.03))
        self.inspectButton.bind(on_press=self.toggleInspect)
        self.add_widget(self.inspectButton)
    def setIcons(self):
        if self.localPlayer is None:
            # Local Player must be set if setting icons
            return
        self.iconTiles = {}
        for T in self.gridtiles.values():
            if (T.tile in cities) and (T.tile != self.localPlayer.birthcity):
                T.lockIcon = LockIcon(T, city_info[T.tile]['entry'])
                T.skirmishIcon = SkirmishIcon(T, 0)
                self.iconTiles[T.tile] = T
                self.add_widget(T.lockIcon)
                self.add_widget(T.skirmishIcon)
    def checkCityUnlocks(self):
        for city, T in self.iconTiles.items():
            if self.localPlayer.Reputation >= city_info[city]['entry']:
                T.lockIcon.delete()
    def updateSkirmishIcons(self):
        sk_cities = set()
        for sk in Skirmishes[0]:
            if (self.localPlayer.birthcity in sk):
                sk_city = list(sk.difference({self.localPlayer.birthcity}))[0]
                sk_cities.add(sk_city)
                self.iconTiles[sk_city].skirmishIcon.set_stage(Skirmishes[0][sk])
        non_sk_cities = set(cities).difference(sk_cities)
        non_sk_cities.remove(self.localPlayer.birthcity)
        for city in non_sk_cities:
            self.iconTiles[city].skirmishIcon.set_stage(0)
    def completeRandomTileDistribution(self):
        np.random.seed(seed)
        randomChoice = np.random.choice(randoms*int(np.ceil(len(positions['randoms'])/len(randoms))), len(positions['randoms']))
        for i in range(len(positions['randoms'])):
            x, y = positions['randoms'][i]
            self.add_tile(randomChoice[i], x, y)
    def semiRandomTileDistribution(self):
        np.random.seed(seed)
        randQ = randoms*int(np.ceil(len(positions['randoms'])/len(randoms)))
        coordQ = deepcopy(positions['randoms'])
        coord = coordQ.pop()
        while randQ:
            nxt = randQ.pop(0)
            same = 0
            neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1 if coord[1] % 2 else -1, 1], [1 if coord[1] % 2 else -1, -1]]
            for neighbor in neighbors:
                n = (coord[0]+neighbor[0], coord[1]+neighbor[1])
                if (n in self.gridtiles) and (self.gridtiles[n].tile == nxt):
                    same += 1
            if np.random.rand() < ((1/10) ** same):
                self.add_tile(nxt, coord[0], coord[1])
                if len(coordQ) == 0:
                    break
                coord = coordQ.pop()
            else:
                randQ.append(nxt)
    def add_tile(self, tiletype, x, y):
        T = Tile(tiletype, x, y)
        self.gridtiles[(x,y)] = T
        T.parentBoard = self
        self.add_widget(T)
    def startRoundDelay(self, delay=None, _=None):
        if delay is None:
            delay = self.defaultRoundDelay
        if delay == 0:
            self.startRound()
        else:
            #msg = f'New Round Starting in {delay}...' if delay == self.defaultRoundDelay else f'{delay}...'
            #self.sendFaceMessage(msg, None, 1, False)
            Clocked(partial(self.startRoundDelay, delay-1), 1, 'start round delay')
    def startRound(self):
        self.localPlayer.savePlayer()
        logging.info("Player saved.")
        for P in self.Players.values():
            P.paused = False
            P.round_ended = False
        if self.localPlayer.dueling_hiatus > 0:
            self.localPlayer.dueling_hiatus -= 1
        for username, value in self.localPlayer.player_fight_hiatus.items():
            if value > 0:
                self.localPlayer.player_fight_hiatus[username] -= 1
        self.localPlayer.unsellable = set()
        self.localPlayer.actions = self.localPlayer.max_actions
        # Update any Quest specs
        if (self.localPlayer.PlayerTrack.Quest.quests[2, 8].status=='started') and hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds'):
            if P.PlayerTrack.Quest.quests[2, 8].wait_rounds > 0:
                P.PlayerTrack.Quest.quests[2, 8].wait_rounds -= 1
        self.localPlayer.round_num += 1
        self.sendFaceMessage(f'Start Round {self.localPlayer.round_num}!', 'cyan')
        if self.localPlayer.has_warrior:
            self.localPlayer.has_warrior -= 1
            if self.localPlayer.has_warrior == 0:
                self.localPlayer.group.pop("Warrior")
                output("Warrior left your group!", 'yellow')
        # Update Tile properties
        for T in self.gridtiles.values():
            T.update_tile_properties()
        if self.localPlayer.currenttile.tile == self.localPlayer.birthcity:
            self.localPlayer.titles['loyal']['in_birthcity'] = True
        if self.localPlayer.fatigue == 0:
            self.localPlayer.updateTitleValue('steady', 1)
        self.localPlayer.titles['decisive']['startRound'] = time()
        # Update Investments
        self.localPlayer.receiveInvestments() # Receive village investments (or at least update)
        # Update stat pages
        self.localPlayer.update_mainStatPage()
        # Check if paralyzed
        paralyzed = check_paralysis()
        if not paralyzed:
            logging.debug("Player not paralyzed.")
            self.localPlayer.started_round = True
            if (self.localPlayer.PlayerTrack.Quest.quests[3, 6].status=='started') and (self.localPlayer.PlayerTrack.Quest.quests[3, 6].action % 2):
                JoinFight()
            elif (self.localPlayer.working[0] is not None):
                self.localPlayer.started_round = False
                perform_labor()
            else:
                self.localPlayer.go2consequence(0)
    def update_market(self, market_seed, _=None):
        np.random.seed(market_seed)
        logging.debug(f"Using seed {market_seed} for market update.")
        for city, D in city_info.items():
            # Update Market
            self.citytiles[city].city_wares = set(np.random.choice(list(D['sell']), 6))
            # Update Jobs
            s = set()
            for skill in skill_users:
                val = round(city_info[city][skill]*5/8) if (skill in city_info[city]) and (city_info[city][skill]>=8) else 2
                if rbtwn(1, 10) <= val:
                    s.add(skill)
            self.citytiles[city].city_jobs = s
    def add_player(self, username, birthcity):
        logging.debug(f"Adding {username} to {birthcity}")
        self.Players[username] = Player(self, username, birthcity)
        self.add_widget(self.Players[username])
        if username == self.localuser:
            # add local player
            self.localPlayer = self.Players[username]
            # So that the start round will appear above the player
            self.startLabel = Label(text='',bold=True,color=(0.5, 0, 1, 0.7),pos_hint={'x':0,'y':0},size_hint=(1,1),font_size=50,markup=True)
            self.add_widget(self.startLabel)
            #self.game_page.recover_button.set_action_func(self.localPlayer.recover)
            self.game_page.recover_button.bind(on_press=self.localPlayer.recover)
            self.game_page.recover_func = self.localPlayer.recover
            self.game_page.actionFuncs['|REST|'] = self.localPlayer.recover
            #self.game_page.eat_button.bind(on_press=self.localPlayer.eat())
            logging.debug("Adding MainStatPage")
            self.localPlayer.add_mainStatPage()
            logging.debug("Adding PlayerTrack")
            self.localPlayer.add_PlayerTrack()
            self.localPlayer.PlayerTrack.craftingTable.update_lbls()

            # Fix city trainers (done at this stage for purposes of Attack/Technique matching with user)
            for city, T in self.citytiles.items():
                adept, master = set(), set()
                for ability, lvl in city_info[city].items():
                    if ability in {'sell', 'entry'}:
                        continue
                    if ability in {'Physical', 'Elemental', 'Wizard', 'Trooper'}:
                        if self.localPlayer.combatstyle != ability:
                            continue
                        # Meaning the teacher can teach the localPlayer Attack or Technique because their combat style matches
                        ability = {'Attack', 'Technique'}
                    else:
                        ability = set([ability])
                    if lvl >= 12:
                        master = master.union(ability)
                        adept = adept.union(ability)
                    elif lvl >= 8:
                        adept = adept.union(ability)
                T.adept_trainers = adept
                T.master_trainers = master

            # Check to see if we are loading the player
            if os.path.exists(f'saves\\{username}\\{load_file}.pickle'):
                logging.debug("Attempting to Load Player")
                self.localPlayer.loadPlayer()
                logging.debug("Load Player Complete")

            # Begin city action loop
            logging.debug("Begin City Action Loop")
            self.localPlayer.go2action()

            # Adjust the screen size
            #width, height = GetSystemMetrics(0), GetSystemMetrics(1)
            # Assumption: Width if greater than height.
            if auto_resize:
                monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
                _, _, width, fullheight = monitor_info.get("Monitor")
                _, _, width, wheight = monitor_info.get("Work") # Account for the taskbar
                wwidth = (1 + self.game_page.right_line_x)*xsize * wheight / ysize # use 1.25 to account for the right_line widget add on
                Window.size = (wwidth, wheight)
                Window.top = 0
                Window.left = 0
                self.game_width = wwidth/(1+self.game_page.right_line_x)
                self.game_height = wheight
                resize_images(self.game_width, self.game_height)

            # Add icon labels to city
            self.setIcons()
    def sendFaceMessage(self, msg=None, clr=None, scheduleTime=2, outputOnMsgBoard=True):
        timeNow = time()
        msg = f'[color={trns_clr[clr]}]{msg}[/color]' if clr is not None else msg
        if msg is not None:
            self.startLblMsgs.append(msg)
            self.startLblTimes.append(timeNow)
            self.startLblLast.append(scheduleTime)
        dlt_i = []
        for i in range(len(self.startLblTimes)):
            if (timeNow - self.startLblTimes[i]) > self.startLblLast[i]:
                dlt_i.append(i)
        self.startLblMsgs = list(np.delete(self.startLblMsgs, dlt_i))
        self.startLblTimes = list(np.delete(self.startLblTimes, dlt_i))
        self.startLblLast = list(np.delete(self.startLblLast, dlt_i))
        self.startLabel.text = '\n'.join(self.startLblMsgs)
        if msg is not None:
            if msg == f'Start Round {self.localPlayer.round_num}!': msg = '\n'+msg
            if outputOnMsgBoard: output(msg, clr)
        def clear_lbl(_):
            self.sendFaceMessage()
        Clocked(clear_lbl, scheduleTime)
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
    def toggleInspect(self, _=None):
        self.inspect_mode = not self.inspect_mode
        self.inspectButton.text = 'Take Action' if self.inspect_mode else 'Inspect'
        self.game_page.outputscreen.current = 'Inspect' if self.inspect_mode else 'Actions'
        self.game_page.secondscreen.current = 'Consequence' if self.inspect_mode else 'Action Grid'

class GamePage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.debug("Creating GamePage")
        self.cols = 2
        self.button_num = 1
        # Determine how wide the right_line widgets can be:
        if auto_resize:
            monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
            _, _, width, height = monitor_info.get("Work") # Account for the taskbar
        else:
            # This should be more reliable across platforms.
            width, height = GetSystemMetrics(0), GetSystemMetrics(1)
        self.right_line_x = (width * ysize) / (xsize * height) - 1
        #self.right_line_x = 0.25
        # Add the Main Screen
        self.main_screen = ScreenManager()
        logging.debug("Creating BoardPage")
        self.board_page = BoardPage(self)
        logging.debug("BoardPage Created Successfully")
        self.city_page = None
        screen = Screen(name='Board')
        screen.add_widget(self.board_page)
        self.main_screen.add_widget(screen)
        self.add_widget(self.main_screen)
        # Add the Side Screen
        input_y, label_y, stat_y, action_y, output_y, toggle_y = 0.05, 0.25, 0.1, 0.2, 0.35, 0.05
        self.stat_ypos, self.stat_ysize = input_y+label_y, stat_y
        self.right_line = RelativeLayout(size_hint_x=self.right_line_x)
        self.hclr = essf.get_hexcolor((255, 85, 0))
        self.vpView = Button(text=f'[color={self.hclr}]3[/color] VP',pos_hint={'x':0, 'y':(input_y+label_y+stat_y+action_y+output_y)},size_hint=(0.2,toggle_y),markup=True,background_color=(0.945, 0.835, 0.475, 1))
        self.vpView.bind(on_press=self.toggleVPHidden)
        self.toggleView = Button(text="Player Track",pos_hint={'x':0.2,'y':(input_y+label_y+stat_y+action_y+output_y)},size_hint=(0.7,toggle_y),background_color=(0, 0.4, 0.4, 1))
        self.toggleView.bind(on_press=self.switchView)
        self.flagButton = Button(text="Flag",pos_hint={'x':0.9,'y':(input_y+label_y+stat_y+action_y+output_y)},size_hint=(0.1,toggle_y),background_color=(0.7, 0, 0, 1))
        self.flagButton.bind(on_press=self.flagPosition)
        self.outputscreen = ScreenManager()
        screen = Screen(name='Actions')
        self.output = ScrollLabel(text='',pos_hint={'x':0,'y':(input_y+label_y+stat_y+action_y)},size_hint=(1,output_y),color=(0.1,0.1,0.1,0.8),valign='bottom',halign='left',markup=True)
        self.output.text_size = self.output.size
        self.output.add_msg('Action Buttons:')
        with self.output.canvas.before:
            Color(0.6, 0.6, 0.8, 0.7, mode='rgba')
            self.outrect=Rectangle(pos=self.output.pos, size=self.output.size)
        self.output.bind(pos=self.update_bkgSize, size=self.update_bkgSize)
        screen.add_widget(self.output)
        self.outputscreen.add_widget(screen)
        screen = Screen(name='Inspect')
        self.inspectView = Table(header = ['Excavate For', 'Base\nProbability', 'Your Max\nProbability'], data=[], header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True, pos_hint={'x':0,'y':(input_y+label_y+stat_y+action_y)},size_hint=(1,output_y))
        screen.add_widget(self.inspectView)
        self.outputscreen.add_widget(screen)

        self.secondscreen = ScreenManager()
        screen = Screen(name='Action Grid')
        logging.debug("Creating actionGrid")
        self.actionGrid = GridLayout(pos_hint={'x':0,'y':(input_y+label_y+stat_y)},size_hint_y=action_y,cols=2)
        self.actionFuncs = {'|REST|': None}
        self.maxHistory = 150 # currently non-adjustable, but allows the capacity to be changed in the future.
        self.actionHistory = []
        self.recover_button = Button(text='Rest (2)')
        self.recover_func = None
        self.occupied = False
        # Button is bound after local player is detected
        self.actionButtons = [self.recover_button]
        screen.add_widget(self.actionGrid)
        logging.debug("Adding actionGrid widget")
        self.secondscreen.add_widget(screen)
        screen = Screen(name='Consequence')
        self.consequenceDisplay = Button(text='', disabled=True, background_disabled_normal='', color=(0,0,0,1), pos_hint={'x':0,'y':(input_y+label_y+stat_y)}, size_hint_y=action_y)
        self.consequenceDisplay.text_size = self.consequenceDisplay.size
        self.consequenceDisplay.bind(pos=self.update_bkgSize, size=self.update_bkgSize)
        screen.add_widget(self.consequenceDisplay)
        self.secondscreen.add_widget(screen)
        self.statGrid = GridLayout(pos_hint={'x':0,'y':(input_y+label_y)},size_hint_y=stat_y,cols=4)
        self.display_page = ScrollLabel(text='',pos_hint={'x':0,'y':input_y},color=(1,1,1,0.8),size_hint=(1,label_y),markup=True)
        self.display_page.text_size = self.display_page.size
        self.display_page.add_msg('Chat Box (press enter to send message)\n')
        with self.display_page.canvas.before:
            Color(0, 0, 0.1, 0.7, mode='rgba')
            self.rect=Rectangle(pos=self.display_page.pos,size=self.display_page.size)
        # Make the display background update itself evertime there is a window size change
        self.display_page.bind(pos=self.update_bkgSize,size=self.update_bkgSize)
        self.new_message = TextInput(pos_hint={'x':0,'y':0},size_hint=(1,input_y))
        # Append the widgets
        self.right_line.add_widget(self.vpView)
        self.right_line.add_widget(self.toggleView)
        self.right_line.add_widget(self.flagButton)
        self.right_line.add_widget(self.outputscreen)
        self.right_line.add_widget(self.secondscreen)
        self.right_line.add_widget(self.display_page)
        self.right_line.add_widget(self.new_message)
        self.add_widget(self.right_line)
        # Any keyboard press will trigger the event:
        Window.bind(on_key_down=self.on_key_down)
    def toggleVPHidden(self, instance):
        if self.vpView.text == 'VP Hidden':
            self.vpView.text = f'[color={self.hclr}]{self.board_page.localPlayer.TotalVP}[/color] VP'
        else:
            self.vpView.text = 'VP Hidden'
    def flagPosition(self, instance):
        if game_launched[0]: logging.warning("User RED FLAG!")
    def switchView(self, instance):
        if self.main_screen.current not in {'Battle', 'Trade'}:
            self.main_screen.current = self.toggleView.text
            self.toggleView.text = "Board" if self.toggleView.text=="Player Track" else "Player Track"
    def make_actionGrid(self, funcDict, save_rest=False, occupied=True, append=False, add_back=False):
        if add_back:
            funcDict['Back'] = self.revert_actionGrid
        self.clear_actionGrid(save_rest, occupied, append)
        for txt, func in funcDict.items():
            orig_txt = deepcopy(txt)
            if txt[0] == '*':
                split = txt.split('|')
                clr, txt = split[0], '|'.join(split[1:])
                B = Button(text=txt, markup=True, color=var.action_color_map[clr]['text'], background_normal='', background_color=var.action_color_map[clr]['background'])
            else:
                B = Button(text=txt, markup=True)
            B.bind(on_press=func)
            self.actionFuncs[orig_txt] = func
            self.actionButtons.append(B)
            self.actionGrid.add_widget(B)
    def revert_actionGrid(self, _=None):
        if len(self.actionHistory) > 0:
            previous_grid = self.actionHistory.pop()
            self.make_actionGrid(previous_grid, append=False)
        else:
            output('Something went wrong, there is no action history to go back too!', 'yellow')
    def get_actionGrid(self):
        return self.actionFuncs
    def check_players(self):
        if (not self.occupied) and (self.board_page.localPlayer is not None):
            Bs = []
            for user in self.board_page.Players:
                if user == self.board_page.localPlayer.username:
                    continue
                elif self.board_page.Players[user].currenttile == self.board_page.localPlayer.currenttile:
                    def view_player(user, _=None):
                        actionGrid({'Trade/Train': partial(player_ask_trade, user),
                                    'Back': exitActionLoop(amt=0)}, False, False)
                    func = partial(view_player, user)
                    B = Button(text=user)
                    B.bind(on_press=func)
                    self.actionFuncs[user] = func
                    Bs.append(B)
            self.actionButtons += Bs
            for B in Bs:
                self.actionGrid.add_widget(B)
    def clear_actionGrid(self, save_rest=False, occupied=True, append=True):
        if append:
            self.actionHistory.append(deepcopy(self.actionFuncs))
            if len(self.actionHistory) > self.maxHistory:
                self.actionHistory.pop(0)
        self.actionFuncs = {}
        for B in self.actionButtons:
            self.actionGrid.remove_widget(B)
        self.occupied = True if (not save_rest) and occupied else False
        if save_rest:
            self.actionButtons = [self.recover_button]
            self.actionFuncs = {'|REST|': self.recover_func}
            self.actionGrid.add_widget(self.recover_button)
            self.check_players()
        else:
            self.actionButtons = []
    def update_bkgSize(self, instance, value):
        self.rect.size = self.display_page.size
        self.rect.pos = self.display_page.pos
        self.outrect.size = self.output.size
        self.outrect.pos = self.output.pos
        self.display_page.text_size = self.display_page.size
        self.output.text_size = self.output.size
        self.consequenceDisplay.text_size = self.consequenceDisplay.size
    def update_output(self, message, color=None):
        if color is None:
            message = message
        elif color in trns_clr:
            message = '[color='+trns_clr[color]+']'+message+'[/color]'
        elif (type(color) is str) and (len(color) == 6):
            message = f'[color={color.lower()}]{message}[/color]'
        else:
            hx = essf.get_hexcolor(color)
            message = '[color='+hx+']'+message+'[/color]'
        self.output.add_msg(message)
    def update_display(self, username, message):
        clr = essf.get_hexcolor((131,215,190)) if username == self.board_page.localuser else essf.get_hexcolor((211, 131, 131))
        #self.display_page.text += f'\n|[color={clr}]{username}[/color]| ' + message
        self.display_page.add_msg(f'|[color={clr}]{username}[/color]| ' + message)
    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        # We want to take an action only when Enter key is being pressed, and send a message
        if keycode == 40:
            # Send Message
            message = self.new_message.text.replace('\n','')
            self.new_message.text = ''
            if (message[0] == '/') and (game_app.launch_page.cheat_on):
                cheat_command = message.split(' ')
                cheat_success, cheat_failure = "007200", "b53000"
                P = lclPlayer()
                if cheat_command[0] == '/help':
                    valid_commands = ['/help', '/skill <skill> <level_gain>', '/attribute <attribute> <level_gain>', '/add_coins <amt>', '/get_item <item> <amt>', '/complete_quest <stage> <mission>', '/execute <command>']
                    output("Valid Commands:", "blue")
                    for command in valid_commands:
                        output(command, '545454')
                elif cheat_command[0] == '/skill':
                    skill = cheat_command[1].title()
                    level_gain = int(cheat_command[2])
                    if (skill not in P.skills) or (level_gain < 1):
                        output("CHEAT COMMAND /skill failed: check if skill name is valid or make sure level_gain >= 1", cheat_failure)
                    else:
                        output(f"CHEAT COMMAND /skill activated: increasing {skill} by {level_gain} levels!", cheat_success)
                        P.updateSkill(skill, level_gain)
                elif cheat_command[0] == '/attribute':
                    attribute = cheat_command[1].title()
                    level_gain = int(cheat_command[2])
                    if (attribute not in P.attributes) or (level_gain < 1):
                        output("CHEAT COMMAND /attribute failed: check if attribute name is valid or make sure level_gain >= 1", cheat_failure)
                    else:
                        output(f"CHEAT COMMAND /attribute activated: increasing {attribute} by {level_gain} levels!", cheat_success)
                        P.updateAttribute(attribute, level_gain)
                elif cheat_command[0] == '/add_coins':
                    add_coins = int(cheat_command[1])
                    output(f"CHEAT COMMAND /add_coins activated: increasing coins by {add_coins}", cheat_success)
                    P.add_coins(add_coins)
                elif cheat_command[0] == '/get_item':
                    item = cheat_command[1].lower()
                    amt = int(cheat_command[2])
                    if (item not in valid_items) or (amt < 1):
                        output("CHEAT COMMAND /get_item failed: check if item name is valid or make sure the amount is >= 1", cheat_failure)
                    else:
                        output(f"CHEAT COMMAND /get_item activated: adding {amt} {item}", cheat_success)
                        P.addItem(item, amt)
                elif cheat_command[0] == '/complete_quest':
                    quest = (int(cheat_command[1]), int(cheat_command[2]))
                    # First add condition if quest is available
                    if quest not in P.PlayerTrack.Quest.quests:
                        output("CHEAT COMMAND /complete_quest failed: the inputted quest does not exist.")
                    elif P.PlayerTrack.Quest.quests[quest].status == 'completed':
                        output("CHEAT COMMAND /complete_quest ignored: the quest is already complete.")
                    else:
                        output("CHEAT COMMAND /complete_quest activated: updating quest to complete.")
                        P.PlayerTrack.Quest.update_quest_status(quest, 'complete')
                elif cheat_command[0] == '/execute':
                    executable = ' '.join(cheat_command[1:])
                    try:
                        exec(executable)
                        output(f"CHEAT COMMAND /execute activated: executed {executable} without error.", cheat_success)
                    except:
                        output(f"CHEAT COMMAND /execute failed, the command threw an error: {executable}.", cheat_failure)
                else:
                    output(f"CHEAT COMMAND {cheat_command[0]} is not recognized!", cheat_failure)
            elif message:
                self.update_display(self.board_page.localuser, message)
                socket_client.send('[CHAT]',message)
    def inspectTile(self, exc, mx):
        if 1 in exc:
            # This means it is tiered
            newexc = {}
            for i in range(len(exc)):
                for e in exc[i+1]:
                    newexc[f'{i+1}. {e}'] = exc[i+1][e]
            exc = newexc
        max_exc_attempts = self.board_page.localPlayer.skills['Excavating'] + 1
        data = []
        for e in exc:
            probability = (exc[e][1]-exc[e][0]+1) / mx
            max_probability = 1 - ((1 - probability)**max_exc_attempts)
            data.append([{'text':e, 'background_color':(1, 1, 1, 1), 'color':(0, 0, 0, 1), 'disabled':True},
                         {'text':str(int(probability*100))+'%', 'background_color':(1, 1, 1, 1), 'color':(0, 0, 0, 1), 'disabled':True},
                         {'text':str(int(max_probability*100))+'%', 'background_color':(1, 1, 1, 1), 'color':(0, 0, 0, 1), 'disabled':True}])
        self.inspectView.update_data_cells(data)
        
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
        display += '[/color]\nKnowledges: [color=ffa500]'+kstr+'[/color]\nCombat Boosts: [color=ffa500]'+cstr+'[/color]'
        tensions = ', '.join([f'{city[0].upper()+city[1:]} ([color=ff3718]{Skirmishes[1][frozenset([city, self.city])]}[/color])' for city in Skirmishes[2][self.city]])
        display += '\nTensions: '+tensions
        self.parentPage.displaylbl.text = '[color=000000]'+display+'[/color]'
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
        
def notAtWar(skrm):
    s = set()
    for S in skrm:
        for city in S:
            s.add(city)
    return set(cities).difference(s)

#%% Game Launch and Server Communication
class LaunchPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loglevel = 'info'
        self.cheat_on = False
        self.cols = 1
        self.username = game_app.connect_page.username.text
        self.usernames, self.ready = [self.username], {self.username:0}
        superGrid = GridLayout(cols=2,height=Window.size[1]*0.9, size_hint_y=None)
        
        settingsGrid = GridLayout(cols=1)
        settingsGrid.add_widget(Button(text="Local Settings", font_size=16, color=(0.2, 0.2, 0.4, 1), underline=True, disabled=True, background_disabled_normal='', background_color=(0.92, 0.92, 0.92, 1), height=Window.size[1]*0.05, size_hint_y=None))
        playerSettings = GridLayout(cols=2)
        self.auto_resize_lbl = Button(text="Auto Resize:\nTrue", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.auto_resize_input = Button(text="Toggle")
        self.auto_resize_input.bind(on_press=self.auto_resize)
        playerSettings.add_widget(self.auto_resize_lbl)
        playerSettings.add_widget(self.auto_resize_input)
        self.logging_level_lbl = Button(text="Logging Level:\nInfo", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.logging_level_input = Button(text='Change to Debug')
        self.logging_level_input.bind(on_press=self.logging_level_change)
        playerSettings.add_widget(self.logging_level_lbl)
        playerSettings.add_widget(self.logging_level_input)
        
        settingsGrid.add_widget(playerSettings)
        settingsGrid.add_widget(Button(text="Global Settings", font_size=16, color=(0.2, 0.2, 0.4, 1), underline=True, disabled=True, background_disabled_normal='', background_color=(0.92, 0.92, 0.92, 1), height=Window.size[1]*0.05, size_hint_y=None))
        
        gameSettings = GridLayout(cols=2)
        self.npc_difficulty_lbl = Button(text="[color=000000]NPC Difficulty:[/color]\n[color=ffb700]Moderate[/color]", color=(0,0,0,1), disabled=True, background_disabled_normal='',markup=True)
        self.npc_difficulty_input = TextInput()
#        dropdown = DropDown()
#        validArgs = ['Very Easy', 'Easy', 'Moderate', 'Hard', 'Very Hard']
#        for index in range(5):
#            btn = Button(text=validArgs[index], color=(1, 1, 1, 1))
#            btn.bind(on_press=lambda btn: dropdown.select(btn.text))
#            dropdown.add_widget(btn)
#        self.npc_difficulty_input = Button(text='Moderate', color=(1, 1, 1, 1))
#        self.npc_difficulty_input.bind(on_release=dropdown.open)
#        dropdown.bind(on_select=lambda instance, x: setattr(self.npc_difficulty_input, 'text', x))
        
        gameSettings.add_widget(self.npc_difficulty_lbl)
        gameSettings.add_widget(self.npc_difficulty_input)
        self.seed_lbl = Button(text="Seed:\nWARNING: Not Connected!", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.seed_input = TextInput()
        gameSettings.add_widget(self.seed_lbl)
        gameSettings.add_widget(self.seed_input)
        self.load_lbl = Button(text="Load:\nWARNING: Not Connected!", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.load_input = TextInput()
        gameSettings.add_widget(self.load_lbl)
        gameSettings.add_widget(self.load_input)
        self.save_lbl = Button(text="Save:\nWARNING: Not Connected!", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.save_input = TextInput()
        gameSettings.add_widget(self.save_lbl)
        gameSettings.add_widget(self.save_input)
        self.end_lbl = Button(text="Game End:\n2:100", color=(0, 0, 0, 1), disabled=True, background_disabled_normal='')
        self.end_input = TextInput()
        gameSettings.add_widget(self.end_lbl)
        gameSettings.add_widget(self.end_input)
        
        settingsGrid.add_widget(gameSettings)
        self.submitButton = Button(text="Request Changes", height=Window.size[1]*0.05, size_hint_y=None, background_color=(0, 2, 0.9, 1))
        self.submitButton.bind(on_press=self.submitChange)
        settingsGrid.add_widget(self.submitButton)
        
        lobbyGrid = GridLayout(cols=1)
        lobbyGrid.add_widget(Button(text="Lobby", font_size=16, color=(0.2, 0.2, 0.4, 1), underline=True, disabled=True, background_disabled_normal='', background_color=(0.96, 1, 1, 1), height=Window.size[1]*0.05, size_hint_y=None))
        self.label = Button(text=f'[color=000000]{self.username}: Not Ready[/color]', markup=True, disabled=True, background_disabled_normal='', background_color=(0.96, 1, 1, 1))#, height=Window.size[1]*0.9, size_hint_y=None
        lobbyGrid.add_widget(self.label)
        superGrid.add_widget(lobbyGrid)
        superGrid.add_widget(settingsGrid)
        self.add_widget(superGrid)
        self.readyButton = Button(text= "Play Single Player")
        self.readyButton.bind(on_press=self.update_self)
        self.add_widget(self.readyButton)
        socket_client.start_listening(self.incoming_message, show_error)
        socket_client.send("[LAUNCH]","Listening")
    def check_launch(self, _=None):
        all_ready = True
        for value in self.ready.values():
            if value == 0:
                all_ready = False
                break
        if all_ready:
            # Create GamePage here to take advantage of seed.
#            game_app.game_page = GamePage()
#            screen = Screen(name='Game')
#            screen.add_widget(game_app.game_page)
#            game_app.screen_manager.add_widget(screen)
            game_app.start_game_page()
            
            #def bootup(_=None):
            game_launched[0] = True
            if not os.path.exists(f'log\\{self.username}'):
                os.makedirs(f'log\\{self.username}')
            game_launched.append(Logger(f'log\\{self.username}\\{save_file}.log'))
            game_launched[1].logger.setLevel(logging.DEBUG if self.loglevel=='debug' else logging.INFO)
            game_launched[1].logger.info("Initializing Logger")
            if os.path.exists(f'saves\\{self.username}\\{load_file}.pickle'):
                with open(f'saves\\{self.username}\\{load_file}.pickle', 'rb') as f:
                    playerInfo = pickle.load(f)
                socket_client.send("[CLAIM]",playerInfo['birthcity'])
                game_app.chooseCity_page.make_claim(self.username, playerInfo['birthcity'])
                game_app.screen_manager.current = 'Game'
            else:
                game_app.screen_manager.current = 'Birth City Chooser'
            #Clocked(bootup, 5)
    def refresh_label(self, _=None):
        self.label.text = '[color=000000]'+'\n'.join([self.usernames[i]+': '+['Not Ready', 'Ready'][self.ready[self.usernames[i]]] for i in range(len(self.usernames))])+'[/color]'
        self.check_launch()
        #Clocked(self.check_launch, 0.5)
    def auto_resize(self, instance):
        global auto_resize
        if self.auto_resize_lbl.text == "Auto Resize:\nTrue":
            self.auto_resize_lbl.text = "Auto Resize:\nFalse"
            auto_resize = False
        else:
            self.auto_resize_lbl.text = "Auto Resize:\nTrue"
            auto_resize = True
    def logging_level_change(self, instance):
        global logging
        if self.logging_level_lbl.text == "Logging Level:\nInfo":
            self.logging_level_lbl.text = "Logging Level:\nDebug"
            self.logging_level_input.text = "Change to Info"
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Logger switching to DEBUG mode")
            self.loglevel = 'debug'
        else:
            self.logging_level_lbl.text = "Logging Level:\nInfo"
            self.logging_level_input.text = "Change to Debug"
            logging.getLogger().setLevel(logging.INFO)
            logging.info("Logger switching to INFO mode")
            self.loglevel = 'info'
    def npc_difficulty(self, new=None):
        global npcDifficulty
        new = self.npc_difficulty_input.text if new is None else new
        validArgs = {'very easy':['Very Easy','[color=43dc00]',1], 'easy':['Easy','[color=438700]',2], 'moderate':['Moderate','[color=ffb700]',3], 'hard':['Hard','[color=ff0700]',4], 'very hard':['Very Hard','[color=850700]',5]}
        if new != '':
            self.npc_difficulty_input.text = ''
            if new.lower() in validArgs:
                self.npc_difficulty_lbl.text = f'[color=000000]NPC Difficulty:[/color]\n{validArgs[new.lower()][1]}{validArgs[new.lower()][0]}[/color]'
                npcDifficulty = validArgs[new.lower()][2]
                self.cheat_on = False
                if self.ready[self.username]:
                    self.update_self()
                return new
            elif new.lower() == 'cheat':
                reverseMap = {1: 'very easy', 2: 'easy', 3: 'moderate', 4: 'hard', 5: 'very hard'}
                self.npc_difficulty_lbl.text = f'[color=000000]NPC Difficulty:[/color]\n{validArgs[reverseMap[npcDifficulty]][1]}{validArgs[reverseMap[npcDifficulty]][0]}[/color] [color=0000b5]|CHEATS ENABLED|[/color]'
                self.cheat_on = True
                if self.ready[self.username]:
                    self.update_self()
                return new
        return False
    def seed(self, new=None):
        global seed
        new = self.seed_input.text if new is None else new
        if new != '':
            self.seed_input.text = ''
            try:
                seed = int(new)
                self.seed_lbl.text = f'Seed:\n{seed}'
                if self.ready[self.username]:
                    self.update_self()
                return seed
            except ValueError:
                pass
        return False
    def load(self, new=None, send=False):
        global load_file
        new = self.load_input.text if new is None else new
        if new != '':
            out = False
            self.load_input.text = ''
            load_file = new
            self.load_lbl.text = f'Load:\n{load_file}'
            if load_file == 'None':
                self.load_lbl.background_color = (1, 1, 1, 1)
            elif os.path.exists(f'saves\\{self.username}\\{load_file}.pickle'):
                # If loading, then the board has to be the same, so receive seed data -- this will create issues if the files are on different games
                self.load_lbl.background_color = (0.8, 1, 0.8, 1)
                with open(f'saves\\{self.username}\\{load_file}.pickle', 'rb') as f:
                    playerInfo = pickle.load(f)
                if send: socket_client.send('[LOAD]', load_file)
                d = self.seed(playerInfo['seed'])
                if d and send: socket_client.send('[SEED]', d)
                s = self.save(load_file)
                if s and send: socket_client.send('[SAVE]', s)
                e = self.gameEnd(playerInfo['gameEnd'])
                if e and send: socket_client.send('[END SETTING]', e)
                if send: socket_client.send('[LOAD SKIRMISHES]', playerInfo['skirmishes'])
                out = load_file
            else:
                self.load_lbl.background_color = (1, 0.8, 0.8, 1)
            if self.ready[self.username]:
                self.update_self()
            return out
        return False
        # [INCOMPLETE] Need to check each player to see if they have the correct data saved
    def save(self, new=None):
        global save_file
        new = self.save_input.text if new is None else new
        if new != '':
            self.save_input.text = ''
            save_file = new
            self.save_lbl.text = f'Save:\n{save_file}'
            if self.ready[self.username]:
                self.update_self()
            return save_file
        return False
    def gameEnd(self, new=None):
        global gameEnd
        new = self.end_input.text if new is None else new
        if new != '':
            valid = True
            params = new.replace(' ','').replace('=',':').split(',')
            potential, texts = {}, []
            for p in params:
                psplit = p.split(':')
                if len(psplit) != 2:
                    valid = False
                    break
                setting, spec = psplit
                if not essf.isint(spec):
                    valid = False
                    break
                spec = int(spec)
                if (int(spec) > 120) or (int(spec) < 0):
                    valid = False
                    break
                if essf.isint(setting) and (int(setting) <= 4) and (int(setting) >= 1):
                    setting = int(setting)
                elif setting.lower() in {'reputation', 'combat', 'capital', 'knowledge'}:
                    setting = setting.lower()[0].upper()+setting.lower()[1:]
                else:
                    valid = False
                    break
                potential[setting] = spec
                texts.append(f'{setting}:{spec}')
            if valid:
                gameEnd = potential
                self.end_lbl.text = 'Game End:\n'+', '.join(texts)
                if self.ready[self.username]:
                    self.update_self()
                return new
        return False
    def submitChange(self, instance):
        n = self.npc_difficulty()
        if n: socket_client.send('[DIFFICULTY]', n)
        l = self.load(send=True)
        if not l:
            d = self.seed()
            if d: socket_client.send('[SEED]', d)
            s = self.save()
            if s: socket_client.send('[SAVE]', s)
            e = self.gameEnd()
            if e: socket_client.send('[END SETTING]', e)
        else:
            self.seed_input.text = ''
            self.save_input.text = ''
            self.end_input.text = ''
    def LAUNCH(self, username, message):
        if username not in self.ready:
            self.usernames.append(username)
            if self.readyButton.text == 'Play Single Player': self.readyButton.text = 'Ready Up'
        self.ready[username] = 1 if message == 'Ready' else 0
        #self.refresh_label()
        Clocked(self.refresh_label, 0.5, 'LAUNCH refresh_label')
    def DIFFICULTY(self, username, message):
        self.npc_difficulty(message)
    def SEED(self, username, message):
        self.seed(message)
    def LOAD(self, username, message):
        self.load(message, send=False)
    def SAVE(self, username, message):
        self.save(message)
    def END_SETTING(self, username, message):
        self.gameEnd(message)
    def CONNECTION(self, username, message):
        if message == 'Closed':
            self.usernames.remove(username)
            self.ready.pop(username)
            output(f'{username} Lost Connection!')
            if game_launched[0] == False:
                self.refresh_label()
    def CLAIM(self, username, message):
        output(f"{username} claimed {message}")
        def clockedClaim(_):
            game_app.chooseCity_page.make_claim(username, message)
        #game_app.chooseCity_page.make_claim(username, message)
        Clocked(clockedClaim, 0.1, 'CLAIM clockedClaim')
    def MOVE(self, username, message):
        def clockedMove(_):
            game_app.game_page.board_page.Players[username].moveto(message, False, True)
        Clocked(clockedMove, 0.2, 'MOVE clockedMove')
    def CHAT(self, username, message):
        def clockedChat(_):
            game_app.game_page.update_display(username, message)
        Clocked(clockedChat, 0.2, 'CHAT clockedChat')
    def EMPTY(self, username, message):
        def emptyTile(_):
            game_app.game_page.board_page.gridtiles[message].empty_tile(recvd=True)
        Clocked(emptyTile, 0.2, 'EMPTY emptyTile')
    def ROUND(self, username, message):
        if message == 'end':
            def pauseUser(_):
                game_app.game_page.board_page.Players[username].end_round()
            Clocked(pauseUser, 0.2, 'ROUND pauseUser')
    def TRADER(self, username, message):
        def clockedTrader(_):
            game_app.game_page.board_page.gridtiles[message[0]].trader_appears(recvd=message[1])
        def clockedPurchase(_):
            game_app.game_page.board_page.gridtiles[message[0]].buy_from_trader(message[1], recvd=True)
        if type(message[1]) is set:
            Clocked(clockedTrader, 0.2, 'TRADER clockedTrader')
        elif type(message[1]) is str:
            Clocked(clockedPurchase, 0.1, 'TRADER clockedPurchase')
        elif message[1] == 0:
            def run(_):
                P = lclPlayer()
                P.remove_trade(True)
            Clocked(run, 0.1, 'TRADER run')
    def SKIRMISH(self, username, message):
        def run(_):
            Skirmishes[0] = message
            game_app.game_page.board_page.updateSkirmishIcons()
            # Move getIncome to the end of the round, not the beginning.
            #game_app.game_page.board_page.localPlayer.getIncome()
        Clocked(run, 0.2, 'SKIRMISH run')
    def MARKET(self, username, message):
        def run(msg=None, _=None):
            game_app.game_page.board_page.update_market(msg)
        #Clocked(partial(game_app.game_page.board_page.update_market, message), 0.01, 'MARKET')
        Clocked(partial(run, message), 0.1, 'MARKET run')
    def EFFICIENCY(self, username, message):
        def run(_):
            output(f"{username} increased village output efficiency by 1 for {message}!", 'blue')
            capital_info[message]['efficiency'] += 1
        Clocked(run, 0.1, 'EFFICIENCY run')
    def DISCOUNT(self, username, message):
        def run(_):
            output(f"{username} convinced {message} market leaders to reduce their prices by 1 coin (min=1)!", 'blue')
            capital_info[message]['discount'] += 1
        Clocked(run, 0.1, 'DISCOUNT run')
    def TRADER_ALLOWED(self, username, message):
        def run(_):
            output(f"{username} convinced traders to start appearing in {message}! (1/8 chance)", 'blue')
            capital_info[message]['trader allowed'] = True
        Clocked(run, 0.1, 'TRADER_ALLOWED run')
    def REDUCED_TENSION(self, username, message):
        def run(_):
            output(f"{username} reduced the tensions between {' and '.join(list(message[0]))} by a factor of {message[1]}!", 'blue')
            Skirmishes[1][message[0]] += message[1]
        Clocked(run, 0.1, 'REDUCED_TENSION run')
    def CAPACITY(self, username, message):
        def run(_):
            output(f"{username} increased {message[0]} home capacity by {message[1]}!", 'blue')
            IncreaseCapacity(message[0], message[1])
        Clocked(run, 0.1, 'CAPACITY run')
    def TITLE_CHANGE(self, username, message):
        def run(_):
            lclPlayer().newMaxRecord(message)
        Clocked(run, 0.1, 'TITLE_CHANGE run')
    def TITLE(self, username, message):
        # deprecated
        def run(_):
            lclPlayer().newMaxRecord(message)
        Clocked(run, 0.1, 'TITLE run')
    def FIGHT(self, username, message):
        def run(_):
            P = lclPlayer()
            if (type(message) is dict) and (P.username in message):
                if message[P.username] == 'declare':
                    player_attempt_fight(username, message["Excavating"], message['stats'])
                elif message[P.username] == 'reject':
                    output(f"{username} is currently unable to fight.", 'yellow')
                    exitActionLoop(amt=0)()
                elif message[P.username] == 'evaded':
                    output(f"{username} evaded the fight.", 'yellow')
                    P.player_fight_hiatus[username] = 2
                    exitActionLoop(amt='minor')()
                elif type(message[P.username]) is int:
                    player_fight(username, message['stats'], 0 if message[P.username]==0 else 1)
                elif (type(message[P.username]) is list) and (len(message[P.username])>0):
                    if 'coins' == message[P.username][0]:
                        output(f"Plundered {message[P.username][1]} coin{'s' if message[P.username[1]]>1 else ''}!", 'green')
                        P.coins += message[P.username][1]
                    else:
                        output(f"Plundered {message[P.username][1]} {message[P.username][0]}!", 'green')
                        P.addItem(message[P.username][0], message[P.username][1])
            elif (type(message) is list) and (P.username==message[0]):
                F = P.parentBoard.game_page.fightscreen.children[0]
                if F.fighting and (F.foename==username):
                    if message[1] == 'confirm':
                        F.foeconfirm()
    def TRADE(self, username, message):
        def run(_):
            P = lclPlayer()
            # Request Trade
            if (type(message) is dict) and (P.username in message):
                if message[P.username] == 'ask':
                    player_confirm_trade(username)
                elif message[P.username] == 'reject':
                    output(f"{username} either rejected or is currently unable to trade.", 'yellow')
                    exitActionLoop(amt=0)()
                elif message[P.username] == 'start':
                    player_trade(username)
            # Make sure both players are trading with each other before delivering updated trade
            elif (P is not None) and (P.is_trading == username) and (P.username == message[0]) and (P.parentBoard.game_page.main_screen.current == 'Trade'):
                T = P.parentBoard.game_page.tradescreen.children[0]
                if len(message) == 4:
                    this_username, offer, amt, slot = message
                    if amt == -1:
                        T.remove(slot[0], slot[1], 1, slot)
                    elif type(offer) is tuple:
                        T.add_train(offer[0], offer[1], 1, slot)
                    elif offer == 'coins':
                        T.add_coins(amt, 1, slot)
                    else:
                        T.add_item(offer, 1, slot)
                elif message[1] == 'confirmed':
                    T.other_confirmation()
                elif message[1] == 'decline':
                    T.receive_rejection()
                elif message[1] == 'abort':
                    output(f"{username} aborted the trade!", 'yellow')
                    T.quit_trade(0)
                elif message[1] == 'no actions':
                    T.display_msg(f"{username} is out of actions. Either wait for round to start or quit trade!", 3, (0.7, 0, 0, 1))
                elif message[1] == 'training failure':
                    T.training_failure()
                elif message[1] == 'training success':
                    T.complete_trade()
        Clocked(run, 0.1, 'TRADE run')
    def GAME_END(self, username, message):
        def run(_):
            P = lclPlayer()
            output(f"{username} Triggered End Game!", 'blue')
            socket_client.send('[END STATS]', {'Combat':P.Combat, 'Reputation':P.Reputation, 'Capital':P.Capital, 'Knowledge':P.Knowledge, 'Fellowship':P.Fellowship, 'Titles':P.Titles})
        Clocked(run, 0.1, 'GAME_END run')
    def FINAL_END_STATS(self, username, message):
        def run(_):
            P = lclPlayer()
            P.GameEnd(message)
        Clocked(run, 0.1, 'FINAL_END_STATS run')
    def incoming_message(self, username, category, message):
        funcAttr = category[1:-1].replace(' ','_')
        if hasattr(self, funcAttr):
            logging.info(f'{username} {funcAttr}')
            getattr(self, funcAttr)(username, message)
        else:
            logging.info(f'IGNORED: {username} {category} {message}')
    def update_self(self, _=None):
        self.ready[self.username] = 1 - self.ready[self.username]
        # send the message to the server that they are ready
        if self.ready[self.username]:
            self.label.background_color = (0.3, 1, 0.3, 1)
            socket_client.send("[LAUNCH]","Ready")
            self.readyButton.text = "Not Ready Anymore" 
        else:
            def blue_screen(_=None):
                self.label.background_color = (0.96, 1, 1, 1)
            self.label.background_color = (1, 0.3, 0.3, 1)
            socket_client.send("[LAUNCH]","Not Ready")
            self.readyButton.text = "Play Single Player" if len(self.ready)==1 else "Ready Up"
            Clocked(blue_screen, 0.3, 'update_self blue_screen')
        self.refresh_label()

class Logger:
    def __init__(self, filename):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('[%(lineno)d] %(asctime)s %(levelname)-8s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        fh.setFormatter(formatter)
        # add the handlers to logger
        self.logger.addHandler(fh)

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
        if os.path.isfile("saves\\prev_details.txt"):
            with open("saves\\prev_details.txt","r") as f:
                d = f.read().split(",")
                prev_ip = d[0]
                prev_port = d[1]
                prev_username = d[2]
        else:
            prev_ip, prev_port, prev_username = '','',''
            
        self.add_widget(Label(text='Username:'))
        self.username = TextInput(text=prev_username, multiline=False)
        self.add_widget(self.username)
        
        self.add_widget(Widget())
        playLocally = Button(text="Play Locally", background_color=(1.5, 0.3, 1.5, 1))
        playLocally.bind(on_press=self.play_local)
        self.add_widget(playLocally)
        
        self.add_widget(Label(text='IP:'))  # widget #1, top left
        self.ip = TextInput(text=prev_ip, multiline=False)  # defining self.ip...
        self.add_widget(self.ip) # widget #2, top right

        self.add_widget(Label(text='Port:'))
        self.port = TextInput(text=prev_port, multiline=False)
        self.add_widget(self.port)

        # add our button.
        self.join = Button(text="Play Globally")
        self.join.bind(on_press=self.join_button)
        self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.join)
        
    def play_local(self, instance):
        socket_server.launch_server()
        def connect(_):
            port = socket_server.PORT
            ip = socket_server.IP # 'localhost'
            username = self.username.text
            if not socket_client.connect(ip, port, username, show_error):
                return
            # Create chat page and activate it
            game_app.start_game_screen()
            game_app.screen_manager.current = 'Launcher'
        game_app.info_page.update_info("Launching Game Locally...")
        game_app.screen_manager.current = 'Info'
        Clocked(connect, 1, 'launch local game')

    def join_button(self, instance):
        port = self.port.text
        ip = self.ip.text
        username = self.username.text
        with open("saves\\prev_details.txt","w") as f:
            f.write(f"{ip},{port},{username}")
        # Create info string, update InfoPage with a message and show it
        info = f"Joining {ip}:{port} as {username}"
        game_app.info_page.update_info(info)
        game_app.screen_manager.current = 'Info'
        Clocked(self.connect, 1, 'join game')

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
        
        # Window Title
        #Window.set_title("Magic Sword: The Board Game") # This didn't work.
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
        
    def start_game_page(self):
        Window.clearcolor = (1, 1, 1, 1)
        
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
    Clocked(sys.exit, 10, 'show error and system exit')

if __name__ == "__main__":
    game_app = EpicApp()
    game_app.run()