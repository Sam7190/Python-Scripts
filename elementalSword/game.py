"""
cd Documents\\GitHub\\Python-Scripts\\elementalSword
python game.py

"""

import buildAI as AI
import numpy as np
import pandas as pd
from PIL import Image as pilImage
import os
import sys
import csv
from collections import Counter
from functools import partial # This would have made a lot of nested functions unnecessary! (if I had known about it earlier)
from copy import deepcopy
import socket_client
from time import time
from win32api import GetSystemMetrics, GetMonitorInfo, MonitorFromPoint
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
from kivy.graphics import Color,Rectangle,Ellipse,InstructionGroup
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.core.window import Window

# Tile Resolution and number of tiles in gameboard.
seed = None
auto_resize = True
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
def randofsum(s, n):
    return np.random.multinomial(s,np.ones(n)/n,size=1)[0]
def euc(a, b): 
    return np.linalg.norm(a-b)

class HitBox(Button):
    def __init__(self, fpage, **kwargs):
        super().__init__(**kwargs)
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
        Clock.schedule_once(self.endAttack, self.TimeRemaining)
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
        #B = Button(text=str(len(self.boxes)), font_size=10, bold=True, background_color=(1,1,1,0), background_disabled_normal='', disabled=True, color=(0, 0.3, 0, 1), pos=self.curBox.pos, size=self.curBox.size)
        #self.lbls.append(B)
        #self.fpage.add_widget(B)
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
                Clock.schedule_once(self.npcDefends, 1)
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
            dis2cent = euc(nrmPos, np.array([0.5, 0.5]))
            if vgID > 0:
                for gi in group2boxes[vgID-1]:
                    if not self.fakes[gi]:
                        dis2last = euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
                        break
            else:
                dis2last = dis2cent
            total, dis2group = 0, 0
            for gi in group2boxes[vgID]:
                if gi == i:
                    continue
                total += 1
                dis2group += euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
            if total > 0: dis2group /= total
            opacity = 0.96 if self.fakes[i] else (0.92 - cunning*0.013)
            data = [[foeAgility, cunning, stability, dodging, vgID, gSize, tApplied, dis2last, dis2cent, dis2group, opacity]]
            # Allow for a 1% bias correction to the model to avoid 100% click rates, and allow for reduce click probability as a function of volley number and order count
            click_prob.append(AI.cmdl.predict_proba(data)[:,1][0]*0.99*(1 - min([0.8, (0.08 * vgID/len(group2boxes))*(self.fpage.order_refresh_count // 3)])))
            dodge_prob.append(AI.dmdl.predict_proba(data)[:,1][0]*0.99*(1 - min([0.8, (0.1 * vgID/len(group2boxes))*(self.fpage.order_refresh_count // 3)])))
        for vgID, boxIDs in sorted(group2boxes.items()):
            clicked_fake = True
            while clicked_fake and (len(boxIDs)>0):
                click_ps = [click_prob[i] for i in boxIDs]
                # Choose the box the NPC is most likely to click
                click_i = np.argmax(click_ps)
                i = boxIDs.pop(click_i)
                click_p, dodge_p = click_prob[i], dodge_prob[i]
                print("Click Probability", click_p, self.fakes[i])
                clickType = None
                if np.random.rand() <= click_p:
                    if not self.fakes[i]: clicked_fake = False
                    # This means the AI "clicked" the box - now check to see if they would have "dodged" the box
                    print("Clicked", "Dodge Probability", dodge_p, self.fakes[i])
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
                self.npcClickBox(i, clickType, 0.3 + 0.3*vgID, 0.3)
            for i in boxIDs:
                # Any remaining ids must be fake, so remove them
                self.npcClickBox(i, None, 0.3 + 0.3*vgID, 0.3)
            #Clock.schedule_once(partial(self.removeBox, i, damageNPC), 0.3 + 0.3*vgID)
        Clock.schedule_once(self.fpage.nextAttack, 0.9 + 0.3*vgID)
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
            Clock.schedule_once(partial(self.removeBox, i, dmg), rmvTime)
        Clock.schedule_once(clk, delay)
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
        Atk = np.max(rbtwn(0, self.fpage.foestats[self.fpage.P.attributes['Attack']])) # Attack chosen technique amt of times, then max is chosen
        if Atk > self.fpage.foestats[self.fpage.P.attributes['Stability']]:
            if rbtwn(0,1)==0:
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
            for coord in normalizedCoords:
                shiftedCoord = [coord[0]**0.5 if coord[0]>=0.5 else (1 - (1 - coord[0])**0.5),
                                coord[1]**0.5 if coord[1]>=0.5 else (1 - (1 - coord[1])**0.5)]
                # Arbitrarily give technique 3 chances to take effect and pick largest outcome
                tApplied = np.max(rbtwn(0, technique, max([2, technique//2])))
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
        d2c = euc(npos, np.array([0.5, 0.5]))
        if vg > 0:
            for gi in self.volley_group[vg-1]:
                if not self.live_rects[gi]['fake']:
                    d2p = euc(npos, self.live_rects[gi]['nrmpos'])
                    break
        else:
            d2p = d2c
        d2g, total = 0, 0
        for gi in self.volley_group[vg]:
            if gi == i:
                continue
            d2g += euc(npos, self.live_rects[gi]['nrmpos'])
            total += 1
        if total > 0: d2g /= total
        clicked = 1 if self.live_rects[i]['state'] in {'blocked', 'dodged'} else 0
        dodged = 1 if self.live_rects[i]['state'] == 'dodged' else 0
        with open('log\\DefenseLog.csv', 'a', newline='') as f:
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
        Clock.schedule_once(transition, self.dodgeTime if prevState=='dodge' else self.blockTime)
    def removeBox(self, i, _=None):
        def rmvBox(_=None):
            self.live_rects[i]['removed'] = True
            self.fpage.canvas.after.remove(self.live_rects[i]['rect'])
            if self.fpage.logDef:
                self.add2Log(i)
        if not self.live_rects[i]['removed']:
            Clock.schedule_once(rmvBox, 0.1)
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
    def __init__(self, name, style, lvl, stats, encountered=True, logDef=False, reward=None, consume=None, action_amt=1, foeisNPC=True, background_img=None, consequence=None, **kwargs):
        super().__init__(**kwargs)
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
        self.P.paused = True
        self.fighting = True
        
        # Stats
        if self.logDef:
            self.pstats = npc_stats(rbtwn(3, 120))
            self.foestats = npc_stats(rbtwn(3, 120))
            output(f"Activating Logged Def, Player Lv.{np.sum(self.pstats)}, NPC Lv.{np.sum(self.foestats)}")
        else:
            self.pstats = deepcopy(self.P.current)
            self.foestats = stats
        
        # Foe objects
        self.foename = name
        self.foelvl = lvl
        self.foestyle = style
        self.foecateg = self.foeCategory()
        self.enc = encountered # enc of -1 means the player is triggering the encounter
        # Calculate Disadvantages
        self.p_affected_stats = set()
        self.f_affected_stats = set()
        self.pstats[self.P.attributes["Stability"]] = max([0, self.pstats[self.P.attributes["Stability"]]-self.P.fatigue]) # Account for fatigue
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
            bkgSource = f'images\\resized\\background\\sparringground.png'
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
        msg = f'You are encountered by {self.foename}\n' if encountered else ''
        msg = msg+f"{fightorder}You can't faint but leveling up probability is reduced.\nRun or Fight?" if self.foename == 'Sparring Partner' else msg+f'{fightorder}Run or Fight?'
        self.msgBoard = Button(text=msg, background_color=(1,1,1,0.6), color=(0.3, 0.3, 0.7, 1), background_disabled_normal='', disabled=True, pos_hint={'x':0.32, 'y':0.8}, size_hint=(0.36, 0.1))#, markup=True)
        self.add_widget(self.msgBoard)
        # Run or Fight
        self.runButton = Button(text=f"", background_color=(1, 1, 0, 1), background_normal='', color=(0,0,0,1), pos_hint={'x':0.32, 'y':0}, size_hint=(0.17, 0.09))
        self.set_runFTG(self.foecateg)
        self.runButton.bind(on_press=self.run)
        category = ['Very Weak', 'Weak', 'Match', 'Strong', 'Very Strong']
        self.fightButton = Button(text=f'Fight ({category[self.foecateg]})', background_color=(0.6, 0, 0, 1), background_normal='', color=(1,1,1,1), pos_hint={'x':0.51, 'y':0}, size_hint=(0.17, 0.09))
        self.fightButton.bind(on_press=self.startBattle)
        self.add_widget(self.runButton)
        self.add_widget(self.fightButton)
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
        self.msgBoard.text = ''
        #self.remove_widget(self.runButton)
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
                Clock.schedule_once(self.triggerNext, delay)
        Clock.schedule_once(trigger, delay)
    def triggerNext(self, _=None):
        if self.fightorder[self.order_idx] == self.P.username:
            self.playerAttacks()
        else:
            self.playerDefends()
    def playerAttacks(self):
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
        self.P.parentBoard.game_page.main_screen.current = 'Board'
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
        elif (self.pstats[self.P.attributes['Hit Points']] == 0) and (self.consequence is not None): 
            # They must have fainted and lost the battle
            self.consequence()
    def foeTakesDamage(self):
        if (self.foestats[self.P.attributes['Hit Points']] > 0) and (self.fighting):
            self.foestats[self.P.attributes['Hit Points']] -= 1
            cell = self.statTable.cells[self.foename][self.P.attributes['Hit Points']]
            cell.text = str(self.foestats[self.P.attributes['Hit Points']])
            cell.background_color = (1, 1, 0.2, 0.6)
            if self.foestats[self.P.attributes['Hit Points']] == 0:
                # Get reward
                if not callable(self.reward):
                    for rwd, amt in self.reward.items():
                        if rwd == 'coins':
                            output(f"Rewarded {int(amt)} coin!", 'green')
                            self.P.coins += int(amt)
                        else:
                            # Assumption: If not coins, then must be an item
                            output(f"Rewarded {int(amt)} {rwd}!", 'green')
                            self.P.addItem(rwd, int(amt))
                # Training with Sparring partner does not gaurantee you a level increase like with others.
                levelsup = (self.foecateg/2) ** 2
                if self.foename == 'Sparring Partner': levelsup *= 0.5
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
                        self.endFight()
                else:
                    output(f"You don't level up, level up remainder: {round(self.P.combatxp,2)}", 'yellow')
                    self.endFight()
    def levelup(self, atr, _=None):
        if self.levelsup > 0:
            output(f"Leveling up {atr}!", 'green')
            self.P.updateAttribute(atr)
            self.levelsup -= 1
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
    def prompt_levelup(self):
        self.check_valid_levelup()
        if (len(self.valid_attributes) > 0) and (self.levelsup > 0):
            output(f"Choose your level up attributes! ({self.levelsup} Remaining)", 'blue')
            actionGrid(self.valid_attributes, False)
        else:
            self.endFight()
    def pTakesDamage(self):
        if (self.pstats[self.P.attributes['Hit Points']] > 0) and (self.fighting):
            self.pstats[self.P.attributes['Hit Points']] -= 1
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
        conditions[0] = 'a[hi]==0'
    if ('Attack' in lbls) and ('Stability' in lbls):
        ai, si = lbls.index('Attack'), lbls.index('Stability')
        conditions[1] = 'a[si]>a[ai]'
    if ('Attack' in lbls):
        ai = lbls.index('Attack')
        conditions[2] = 'a[ai]==0'
    if ('Technique' in lbls):
        ti = lbls.index('Technique')
        conditions[3] = 'a[ti]>12'
    if fixed is not None:
        for atr, fixedlvl in fixed.items():
            if atr != 'Stealth':
                conditions.append(f'a[{P.attributes[atr]}]=={fixedlvl}')
            else:
                # [INCOMPLETE] Add a stat for stealth?
                pass
    while True:
        a = randofsum(lvl, n)
        breakout = True
        for i in range(len(conditions)):
            if (conditions[i] is not None) and eval(conditions[i]):
                breakout = False
                break
        if breakout: break
    return a

def encounter(name, lvlrange, styles, reward, fixed=None, party_size=1, consume=None, action_amt=1, empty_tile=False, lvlBalance=6, enc=1, consequence=None, background_img=None):
    P = lclPlayer()
    possibleLvls = rbtwn(lvlrange[0],lvlrange[1],max([1, round((lvlrange[1]-lvlrange[0])/lvlBalance)]))
    lvl = possibleLvls[np.argsort(np.abs(possibleLvls-np.sum(P.current)))[0]]
    # For rewards go through the dictionary and if value is a list then choose a random result biased toward the level chosen of opponent
    if not callable(reward):
        new_reward = {}
        for rwd, val in reward.items():
            if type(val) is list:
                lvlpct = (lvl - lvlrange[0])/(lvlrange[1] - lvlrange[0])
                val = round(lvlpct * (val[1] - val[0]))
            new_reward[rwd] = val
    else:
        new_reward = reward
    cbstyle = styles[rbtwn(0,len(styles)-1)]
    output(f'Encounter Lv{lvl} {name} using {cbstyle}','yellow')
    screen = Screen(name="Battle")
    screen.add_widget(FightPage(name, cbstyle, lvl, npc_stats(lvl, fixed), enc, reward=new_reward, consume=consume, action_amt=action_amt, foeisNPC=True, consequence=consequence, background_img=background_img))
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

def output(message, color=None):
    game_app.game_page.update_output(message, color)
    print(message)
def actionGrid(funcDict, save_rest):
    game_app.game_page.make_actionGrid(funcDict, save_rest)
def check_paralysis():
    P = lclPlayer()
    paralyzed = False
    if (P.fatigue > P.max_fatigue) or (P.paralyzed_rounds in {1,2}):
        paralyzed = True
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
                elif amt>0:
                    P.takeAction(amt)
                if empty_tile: P.currenttile.empty_tile()
                if (consume is None) or ('tiered' not in consume):
                    P.go2action()
                else:
                    # Otherwise the player must be descended into a cave or ontop of a mountain
                    tier = int(consume[-1])
                    P.tiered = tier if tier > 1 else False
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
        actionGrid({'Yes':partial(sellItem, item, to_trader, True), 'No':partial(sellItem, item, to_trader, False), 'Cancel':exitActionLoop(amt=0)}, False)
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
            else:
                P.bartering_mode = 1
                sellprice += 1
                P.coins += sellprice
            # Its basically inverted getItem
            output(f"Sold {item} for {sellprice}.")
            getItem(item, -1, 'minor', False, 1)()
        else:
            output("You failed to barter, sell anyway?", 'yellow')
            actionGrid({'Yes':partial(sellItem, item, to_trader, False), 'No':exitActionLoop('minor', 0, False)}, False)
    else:
        sellprice += P.bartering_mode
        output(f"Sold {item} for {sellprice}.")
        P.coins += sellprice
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
                output(f"Bought {item} for {cost}")
            else:
                output("Bought {amt} {item} for {cost*amt}")
            if from_trader and (item in P.currenttile.trader_wares):
                P.currenttile.buy_from_trader(item)
            getItem(item, amt, consume, False, 1)()
    return buyitem

def Trading(trader, _=None):
    P = lclPlayer()
    if P.paused:
        return
    items = P.currenttile.trader_wares if trader else P.currenttile.city_wares
    if (P.currenttile.tile == 'benfriege') and (P.PlayerTrack.Quest.quests[(2, 1)].status == 'started'):
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
        # Reduce cost depending on trader and bartering mode, but min cost must be 1.
        reduce = P.bartering_mode if trader else min([1, P.bartering_mode])
        cost = max([cost - reduce, 1])
        clr = 'ffff75' if P.bartering_mode == 0 else '00ff75'
        actions[f'{item}:[color={clr}]{cost}[/color]'] = buyItem(item, cost, trader, consume='minor')
        # Give the player the option to go back to regular action menu of the tile
        actions['Back'] = exitActionLoop(None, 0)
    # If the player has not activated bartering already, give them a chance to do so
    if (P.bartering_mode == 0) and (P.activated_bartering==False): actions['Barter'] = activate_bartering
    actionGrid(actions, False)

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
        return
    critthink = P.activateSkill("Critical Thinking")
    r = rbtwn(0, 8, None, critthink, 'Critical Thinking ')
    if r <= critthink:
        P.useSkill("Critical Thinking")
        output(f"You successfully read the {book}!", 'green')
        P.addItem(book, -1)
        P.updateSkill(book2skill(book), 1)
    else:
        output("You failed to understand. You can try again.", 'red')
    exitActionLoop('minor')()

def Train(abilities, master, confirmed, _=None):
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
        output(f"Would you like to train at cost of {cost} coins?", 'blue')
        game_app.game_page.make_actionGrid({'Yes':(lambda _:Train(abilities, master, True)), 'No':exitActionLoop(amt=0 if master in {'adept','city'} else 1)}, False)
    elif type(abilities) is not str:
        output("Which skill would you like to train?", 'blue')
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
                # The assumption is that if they are training in city they did not just move into the tile.
                exitActionLoop(None,0 if master=='city' else 1)()
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
        # [INCOMPLETE] Have trader run away
        stealth = P.activateSkill("Stealth")
        r = rbtwn(1, 12, None, stealth, 'Stealth ')
        if r <= stealth:
            P.useSkill('Stealth')
            output("Successfully avoided robber!", 'green')
            exitActionLoop('road')()
        else:
            output("Unable to avoid robber!", 'yellow')
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
    exc = {'Hunstman':[1,1,persuade_trainer(['Agility','Gathering'],'huntsman',exitActionLoop())],
           'Wild Herd':[2,6,Gather('raw meat',0)]}
    if inspect:
        return exc, 9
    else:
        P = lclPlayer()
        def FindBabyMammoth(_=None):
            output("You found a baby mammoth! Its attracted to your fruit and follows you. Don't lose your fruit!", 'green')
            P.PlayerTrack.Quest.quests[(1, 8)].has_mammoth = True
            exitActionLoop()()
        if (P.PlayerTrack.Quest.quests[(1, 8)].status == 'started') and ('fruit' in P.items):
            exc['Wild Herd'][2] = FindBabyMammoth
        actions = {'Excavate':Excavate(exc,9)}
        actionGrid(actions, True)
    
def A_pond(inspect=False):
    exc = {'Go Fishing':[1,5,Gather('raw fish',0)],
           'Clay':[6,8,getItem('clay')],
           'Giant Serpent':[9,9,partial(encounter,'Giant Serpent',[20,45],['Elemental'],{'scales':[1,3]})]}
    if inspect:
        return exc, 12
    else:
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
        C_mountain(tier+1)
    exc = {1:{'Copper':[1,5,getItem('copper')],'Iron':[6,10,getItem('iron')],'Monk':[11,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           2:{'Kevlium':[1,4,getItem('kevlium')],'Nickel':[5,8,getItem('nickel')],'Monk':[9,11,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]},
           3:{'Diamond':[1,3,getItem('tungsten')],'Chromium':[4,6,getItem('titanium')],'Monk':[7,8,persuade_trainer(['Survival','Excavating'],'monk',exitActionLoop())]}}
    if inspect:
        return exc, 20
    else:
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
    if inspect:
        return exc, 20
    else:
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
    def rare_find(_):
        r = rbtwn(1, 6)
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
    if inspect:
        return exc, 20
    else:
        actions = {'Excavate':Excavate(exc,20)}
        actionGrid(actions, True)

avail_actions = {'road':A_road,'plains':A_plains,'pond':A_pond,'cave':A_cave,'outpost':A_outpost,'mountain':A_mountain,'oldlibrary':A_oldlibrary,'ruins':A_ruins,'battle1':A_battlezone,'battle2':A_battlezone}

food_restore = {'raw meat':(1,0),'cooked meat':(2,0),'well cooked meat':(3,0),
                'raw fish':(0,1),'cooked fish':(0,2),'well cooked fish':(0,3),
                'fruit':(1,1)}

ore_properties = {'lead':('Elemental',1),'tin':('Physical',1),'copper':('Trooper',1),'iron':('Wizard',1),
                  'tantalum':('Elemental',2),'aluminum':('Physical',2),'kevlium':('Trooper',2),'nickel':('Wizard',2),
                  'tungsten':('Elemental',3),'titanium':('Physical',3),'diamond':('Trooper',3),'chromium':('Wizard',3),
                  'shinopsis':('Elemental',4),'ebony':('Physical',4),'astatine':('Trooper',4),'promethium':('Wizard',4)}

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

city_info = {'anafola':{'Hit Points':8, 'Stability':8, 'Wizard':8, 'Persuasion':8, 'Excavating':12, 'Smithing':4, 'entry':12, 'sell':{'raw fish', 'cooked fish', 'string', 'beads', 'sand', 'scales', 'bark', 'lead', 'tin', 'copper',' iron', 'persuasion book'}},
             'benfriege':{'Hit Points':8, 'Stability':8, 'Cunning':8, 'Elemental':8, 'Def-Trooper':8, 'Critical Thinking':8, 'Persuasion':8, 'Crafting':12, 'Survival':8, 'Smithing':4, 'entry':4, 'sell':{'raw fish', 'cooked fish', 'well cooked fish', 'string', 'beads', 'scales', 'bark', 'critical thinking book', 'crafting book', 'survival book', 'gathering book'}},
             'demetry':{'Elemental':8, 'Def-Trooper':8, 'Bartering':12, 'Crafting':8, 'Smithing':4, 'entry':13, 'sell':{'raw meat', 'cooked meat', 'fruit', 'string', 'beads', 'hide', 'sand', 'clay', 'leather', 'ceramic', 'glass', 'gems', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'tungsten', 'bartering book', 'crafting book'}},
             'enfeir':{'Agility':8, 'Cunning':12, 'Physical':8, 'Def-Wizard':8, 'Critical Thinking':8, 'Heating':8, 'Smithing':8, 'Stealth':12, 'entry':3, 'sell':{'raw meat', 'cooked meat', 'string', 'hide', 'tin', 'copper', 'aluminum', 'kevlium'}},
             'fodker':{'Stability':12, 'Trooper':8, 'Def-Physical':12, 'Bartering':8, 'Smithing':8, 'entry':6, 'sell':{'raw meat', 'cooked meat', 'string', 'hide', 'sand', 'lead', 'tin', 'copper', 'iron', 'excavating book'}},
             'glaser':{'Stability':8, 'Cunning':8, 'Elemental':8, 'Def-Trooper':12, 'Critical Thinking':8, 'Persuasion':8, 'Crafting':8, 'Survival':8, 'entry':4, 'Excavating':8, 'Smithing':4, 'sell':{'raw fish', 'cooked fish', 'string', 'beads', 'scales', 'bark', 'lead', 'tin', 'critical thinking book', 'survival book', 'gathering book'}},
             'kubani':{'Agility':8, 'Physical':8, 'Def-Wizard':12, 'Critical Thinking':8, 'Bartering':8, 'Crafting':8, 'Gathering':8, 'Smithing':4, 'entry':6, 'sell':{'raw meat', 'string', 'beads', 'sand', 'clay', 'glass', 'lead', 'copper', 'iron', 'tantalum', 'titanium', 'survival book'}},
             'pafiz':{'Hit Points':8, 'Wizard':8, 'Def-Elemental':12, 'Persuasion':12, 'Crafting':8, 'Heating':8, 'Gathering':8, 'Smithing':4, 'entry':6, 'sell':{'cooked meat', 'cooked fish', 'fruit', 'string', 'beads', 'hide', 'scales', 'iron', 'nickel', 'persuasion book'}},
             'scetcher':{'Hit Points':8, 'Agility':8, 'Stability':8, 'Physical':8, 'Wizard':8, 'Elemental':8, 'Trooper':8, 'Def-Physical':8, 'Def-Wizard':8, 'Def-Elemental':8, 'Def-Trooper':8, 'Smithing':8, 'Stealth':8, 'entry':1, 'sell':{'raw meat', 'raw fish', 'string', 'sand', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'aluminum', 'kevlium', 'nickel'}},
             'starfex':{'Hit Points':8, 'Elemental':12, 'Def-Trooper':8, 'Heating':8, 'Gathering':8, 'Excavating':8, 'Smithing':4, 'entry':8, 'sell':{'raw fish', 'cooked fish', 'fruit', 'string', 'beads', 'scales', 'lead', 'tantalum', 'tungsten', 'heating book'}},
             'tamarania':{'Hit Points':8, 'Stability':8, 'Physical':12, 'Def-Wizard':8, 'Smithing':12, 'entry':14, 'sell':{'raw meat', 'cooked meat', 'well cooked meat', 'string', 'beads', 'hide', 'clay', 'leather', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'aluminum', 'kevlium', 'nickel', 'titanium', 'diamond', 'smithing book'}},
             'tamariza':{'Hit Points':8, 'Cunning':8, 'Wizard':12, 'Def-Elemental':8, 'Critical Thinking':8, 'Persuasion':8, 'Heating':8, 'Smithing':4, 'entry':15, 'sell':{'fruit', 'string', 'beads', 'bark', 'rubber', 'iron', 'nickel', 'chromium', 'heating book'}},
             'tutalu':{'Hit Points':8, 'Trooper':12, 'Def-Physical':8, 'Smtihing':8, 'Excavating':8, 'entry':8, 'sell':{'raw meat', 'cooked meat', 'string', 'beads', 'hide', 'leather', 'copper', 'kevlium', 'diamond', 'excavating book'}},
             'zinzibar':{'Agility':12, 'Physical':8, 'Def-Wizard':8, 'Persuasion':8, 'Smithing':8, 'Survival':12, 'entry':3, 'sell':{'raw meat', 'cooked meat', 'string', 'hide', 'lead', 'tin', 'tantalum', 'aluminum'}}}

village_invest = {'village1':'Food', 'village2':'Crafting', 'village3':'Cloth', 'village4':'Food', 'village5':'Crafting'}

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
Skirmishes = [{}]

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
                

class Player(Image):
    def __init__(self, board, username, birthcity, is_human=True, **kwargs):
        super().__init__(**kwargs)
        self.source = f'images\\characters\\{birthcity}.png'
        self.parentBoard = board
        self.username = username
        self.birthcity = birthcity
        self.is_human = is_human
        self.imgSize = pilImage.open(self.source).size
        
        # Player Victory Points
        self.Combat = 3
        self.Capital = 0
        self.Reputation = 0
        self.Knowledge = 0
        
        # Constraints
        self.paused = False
        self.started_round = False
        self.ate = 0
        self.max_eating = 2
        self.max_road_moves = 2
        self.max_actions = 2
        self.max_minor_actions = 3
        self.max_capacity = 3
        self.max_fatigue = 10
        self.stability_impact = 1
        self.combatstyle = cities[self.birthcity]['Combat Style']
        
        # Current Stats
        self.road_moves = 2
        self.actions = 2
        self.minor_actions = 3
        self.item_count = 0
        self.items = {}
        self.activated_bartering = False
        self.bartering_mode = 0
        self.unsellable = set()
        self.fatigue = 0
        self.dueling_hiatus = 0
        self.free_smithing_rent = False
        self.group = set()
        
        self.coins = cities[self.birthcity]['Coins']
        self.paralyzed_rounds = 0
        self.tiered = False
        
        # Player Track
        #Combat
        self.combatxp = 0
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
    def go2consequence(self, amt=1):
        if self.currenttile.tile in consequences:
            if self.tiered:
                # In the case that they are in the mountain or cave, then make sure they go to the right tier.
                consequences[self.currenttile.tile](self.tiered)
            else:
                consequences[self.currenttile.tile]()
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
    def moveto(self, coord, trigger_consequence=True, skip_check=False):
        nxt = self.parentBoard.gridtiles[coord]
        if nxt.tile in cities:
            # Can't enter the city unless was fainted and rushed to the hospital"
            if (not self.entry_allowed[nxt.tile]) and (self.current[self.attributes["Hit Points"]] > 0):
                output(f"The city will not let you enter! (Need {city_info[nxt.tile]['entry']}+ Reputation!)",'red')
                return
        if (not skip_check) and (self.currenttile.tile != 'road') and (self.currenttile.tile not in cities) and (nxt.tile=='road'):
            output("Move there on same action (+2 Fatigue)?", 'blue')
            actionGrid({'Yes':(lambda _:self.moveto(coord,trigger_consequence,3)),'No':(lambda _:self.moveto(coord,trigger_consequence,2))}, False)
        else:
            if skip_check != True:
                socket_client.send('[MOVE]',coord)
            self.currentcoord = coord
            self.currenttile = self.parentBoard.gridtiles[coord]
            # Collect any money waiting at bank
            if (self.currenttile.tile in cities):
                self.coins += self.bank[self.currenttile.tile]
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
    def update_mainStatPage(self):
        updated_data = self.get_mainStatUpdate()
        for i in range(len(updated_data)):
            self.mtable.cells[self.mtable.header[i]][0].text = updated_data[i]
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
            self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+rest_rate])
            output(f'[ACTION {self.max_actions-self.actions+1}] You rested {rest_rate} fatigue/HP')
            self.takeAction(0, False, True)
    def eat(self, food, _=None):
        if self.paused:
            return
        if self.ate >= self.max_eating:
            output("You can't eat anymore this action!",'yellow')
            return
        self.ate += 1
        ftg, hp = food_restore[food]
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
        if ftg>0: restoring_list.append(f'FTG by {ftg}')
        if hp>0: restoring_list.append(f'HP by {hp}')
        output(f'Restored {", ".join(restoring_list)}','green')
        self.addItem(food, -1)
        self.fatigue = max([0, self.fatigue - ftg])
        hp_idx = self.attributes['Hit Points']
        self.current[hp_idx] = min([self.combat[hp_idx]+self.boosts[hp_idx], self.current[hp_idx]+hp])
        self.update_mainStatPage()
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
        self.parentBoard.startRound()
    def takeDamage(self, hp_amt, ftg_amt, add=True):
        hp_idx = self.attributes['Hit Points']
        self.current[hp_idx] = max([0, self.current[hp_idx]-hp_amt])
        fainted = False
        if self.current[hp_idx] == 0:
            self.parentBoard.sendFaceMessage('You Fainted!','red')
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
            coord, distance = self.currenttile.findNearest(cities)
            self.moveto(coord, trigger_consequence=False)
            # Artificially paralyze the player by setting fatigue 1 greater than max
            self.fatigue = self.max_fatigue + 1
            fainted = True
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
            self.bartering_mode = 0
        if (fatigue > 0) and (self.paralyzed_rounds == 0):
            # Not activated, but can be used.
            survival = self.skills["Survival"]
            r = rbtwn(1, 16, None, survival, 'Survival ') # Adept survival gives 0.5 chance of no fatigue, while masters have 0.75 chance of no fatigue.
            if r <= survival:
                self.useSkill("Survival")
                fatigue -= 1
                output("Your survival skill saved you a fatigue!", 'green')
            if fatigue > 0:
                output("Took normal fatigue this action")
                self.takeDamage(0,fatigue)
        self.minor_actions = self.max_minor_actions # Refresh minor actions
        self.ate = 0 # Refresh how much one can eat
        self.road_moves = self.max_road_moves # If action is taken, then the road moves should be refreshed.
        self.already_asset_bartered = False # If an action is taken, you can try to barter for the asset again.
        self.update_mainStatPage()
        if self.actions <= 0:
            self.pause()
            socket_client.send("[ROUND]",'end')
    def updateSkill(self, skill, val=1):
        self.skills[skill] += val
        self.Knowledge += val
        if (skill == 'Crafting') and (hasattr(self, 'PlayerTrack')):
            self.PlayerTrack.craftingTable.update_lbls()
    def addXP(self, skill, xp, msg=''):
        output(f"{msg}Gained {xp}xp for {skill}.",'green')
        self.xps[skill] += xp
        while (self.xps[skill] >= (3 + self.skills[skill])):
            self.xps[skill] -= (3 + self.skills[skill])
            self.updateSkill(skill)
            output(f"Leveled up {skill} to {self.skills[skill]}!",'green')
    def activateSkill(self, skill, xp=1, max_lvl_xp=2):
        lvl = self.skills[skill]
        if rbtwn(1,self.max_fatigue) <= self.fatigue:
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
    def applyBoost(self, attribute, val=1, update_stats=True):
        index = self.attributes[attribute]
        self.boosts[index] += val
        self.current[index] += val
        if update_stats: self.update_mainStatPage()
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
                exitActionLoop(amt=0)()
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
        for city in notwaring:
            # If you have the city
            if self.markets[city]:
                # And you are on the city
                if self.currenttile.tile == city:
                    # Check if bartering will give you extra coins - does not count as an activation!
                    r = rbtwn(0, self.skills["Bartering"])
                    self.coins += capital_info[city]['return'] + r # Get the income plus the bartering effort
                # Otherwise check if you have automated the city
                elif self.workers[city]:
                    self.bank[city] += capital_info[city]['return'] - 1 # 1 coin goes to the worker and money sent to bank
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
        self.coins -= cost
        exitActionLoop(amt=1)()
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
                    self.villages[city][v][1] = round(rounds + (6 - rounds)/1.9)
                    self.villages[city][v][2] = 1 # Have one waiting after the round count is over
    def addItem(self, item, amt, store=False):
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
                output(f"{item} doesn't exit in the inventory!", 'yellow')
                return
            elif (item == 'fruit') and ((self.items[item] - amt) <= 0) and (self.PlayerTrack.Quest.quests[(1, 8)].status == 'started') and hasattr(self.PlayerTrack.Quest.quests[(1, 8)], 'has_mammoth') and self.PlayerTrack.Quest.quests[(1, 8)].has_mammoth:
                output(f"You lost your fruit! The baby mammoth stops following you! You will need to find it again.", 'red')
                self.PlayerTrack.Quest.quests[(1, 8)].has_mammoth = False
        self.item_count += amt
        if item in self.items:
            self.items[item] += amt
        else:
            self.items[item] = amt
        if self.items[item]==0:
            self.items.pop(item)
        self.update_mainStatPage()

class HoverButton(Button, HoverBehavior):
    def __init__(self, display, message, **kwargs):
        super().__init__(**kwargs)
        self.display=display
        self.message=message
    def on_enter(self, *args):
        self.display.text = self.message

class Table(GridLayout):
    def __init__(self, header, data, wrap_text=True, bkg_color=None, header_color=None, text_color=None, super_header=None, header_height_hint=None, header_as_buttons=False, color_odd_rows=False, header_bkg_color=(1,1,1,0), **kwargs):
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
                L = Button(text=clr[0]+h+clr[1],markup=True,background_color=header_bkg_color,underline=True,**hkwargs)
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
                elif (type(item) is str) or (type(item) is int) or (type(item) is float):
                    L = Label(text=str(item)) if self.input_text_color is None else Label(text=str(item),color=self.input_text_color)
                    if self.wrap_text: L.text_size = L.size
                else:
                    # Assume it is a widget
                    L = item
                self.add_widget(L)
                self.cells[cell].append(L)
            i += 1
    def update_bkgSize(self, instance, value):
        self.bkg.size = self.size
        self.bkg.pos = self.pos
        if self.wrap_text:
            for L in self.children:
                L.text_size = L.size
                
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
                txt, dsbld, clr = ("Equipped",True,(0,0,0,1)) if space==1 else ("",False,(1,1,1,1))
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
            output(f"Would you like to remove and destroy this item?", "blue")
            actionGrid({"Yes":partial(self.rmv_slot, space, slot, True), "No":exitActionLoop(amt=0)}, False)
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
                self.aspace[space][slot].text = item
                self.aspace[space][slot].color = (0, 0.6, 0, 1)
                self.aspace[space][slot].disabled = False
                self.space_items[space] += 1
                self.P.addItem(item, -1)
                self.P.item_count += 1
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
                        'Cancel':exitActionLoop(amt=0)}, False)
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
        elif (self.P.currenttile not in cities) and (self.P.currenttile.trader_rounds==0):
            # If you are not in a city or not in any trading places, then you can't sell the item
            return
        elif (self.space_items[space]==1) and (self.aspace[space][1].text in self.P.unsellable):
            output("You can't sell that item this turn!", 'yellow')
            return
        sellprice = self.sell_value(space)
        if (barter is None) and (not self.P.activated_bartering):
            output("Would you like to barter?",'blue')
            actionGrid({'Yes':partial(self.sell_craft, space, True), 'No':partial(self.sell_craft, space, False), 'Cancel':exitActionLoop(amt=0)}, False)
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
                self.rmv(space, True)
            else:
                output("You failed to barter, sell anyway?", 'yellow')
                actionGrid({'Yes':partial(self.sell, space, False), 'No':exitActionLoop('minor', 0, False)}, False)
        else:
            sellprice += self.P.bartering_mode
            output(f"Sold smithed piece for {sellprice}.")
            self.P.coins += sellprice
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
            output(f"Would you like to remove and destroy this item?", "blue")
            actionGrid({"Yes":partial(self.rmv_slot, space, slot, True), "No":exitActionLoop(amt=0)}, False)
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
    def add_craft(self, item, space=None, _=None):
        if self.P.paused:
            return
        if space is None:
            output("Which craft would you like to add it to?", 'blue')
            actionGrid({'Craft 1':partial(self.add_craft, item, 0), 'Craft 2':partial(self.add_craft, item, 1), 'Cancel':exitActionLoop(amt=0)}, False)
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
                        self.cspace[space][slot].text = item
                        self.cspace[space][slot].color = (0, 0.6, 0, 1)
                        self.cspace[space][slot].disabled = False
                        self.space_items[space] += 1
                        self.P.addItem(item, -1)
                        self.P.item_count += 1
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
        elif (self.P.currenttile not in cities) and (self.P.currenttile.trader_rounds==0):
            # If you are not in a city or not in any trading places, then you can't sell the item
            return
        elif (self.space_items[space]==1) and (self.cspace[space][1].text in self.P.unsellable):
            output("You can't sell that item this turn!", 'yellow')
            return
        sellprice = self.sell_value(space)
        if (barter is None) and (not self.P.activated_bartering):
            output("Would you like to barter?",'blue')
            actionGrid({'Yes':partial(self.sell_craft, space, True), 'No':partial(self.sell_craft, space, False), 'Cancel':exitActionLoop(amt=0)}, False)
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
                self.rmv_craft(space, True)
            else:
                output("You failed to barter, sell anyway?", 'yellow')
                actionGrid({'Yes':partial(self.sell_craft, space, False), 'No':exitActionLoop('minor', 0, False)}, False)
        else:
            sellprice += self.P.bartering_mode
            output(f"Sold craft {space+1} for {sellprice}.")
            self.P.coins += sellprice
            self.rmv_craft(space, True)
            
quest_req = {(1, 3): 'self.playerTrack.player.actions == self.playerTrack.player.max_actions',
             (1, 8): "'fruit' in self.playerTrack.player.items",
             (2, 5): "self.quests[1, 6].status == 'complete'",
             (2, 7): "self.quests[1, 4].status == 'complete'",
             (3, 4): "self.playerTrack.player.Combat >= 40",
             (3, 6): "False", # [INCOMPLETE] Figure out if beginning of skirmish for a city -- then choose that city
             (3, 8): "self.quests[1, 7].status == 'complete'",
             (4, 1): "self.quests[2, 5].status == 'complete'",
             (4, 7): "(self.quests[1, 2].status == 'complete') and (self.playerTrack.player.skills['Crafting'] >= 6)",
             (5, 5): "self.playerTrack.player.coins >= 40",
             (5, 6): "self.quests[1, 8].status == 'complete'",
             (5, 7): "self.quests[2, 8].status == 'complete'",
             (5, 8): "self.playerTrack.player.skills['Crafting'] >= 12"}

def getQuest(stage, mission):
    P = lclPlayer()
    return P.PlayerTrack.Quest.quests[(stage, mission)]

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
        P.addItem('string')
        P.addItem('beads')
        P.PlayerTrack.Quest.update_quest_status((1, 2), 'complete')
    exitActionLoop()()
    
def GaurdHome(_=None):
    P = lclPlayer()
    if P.paused:
        return
    B = getQuest(1, 3)
    def Reward():
        actions = {'Tin': getItem('tin', 1, action_amt=0),
                   'Iron': getItem('iron', 1, action_amt=0),
                   'Lead': getItem('lead', 1, action_amt=0),
                   'Copper': getItem('copper', 1, action_amt=0)}
        P.PlayerTrack.Quest.update_quest_status((1, 3), 'complete')
        output("The owner wants to reward you for gaurding the house! Choose one of the following:", 'blue')
        actionGrid(actions, False)
    def Consequence():
        P.PlayerTrack.Quest.update_quest_status((1, 3), 'failed')
    B.count = B.count + 1 if hasattr(B, 'count') else 1
    output(f"You spent an action gaurding the house")
    if B.count >= 2:
        encounter('Robber', [6, 6], ['Physical'], Reward, consequence=Consequence, background_img='images\\resized\\background\\cottage.png')
    else:
        exitActionLoop()()
        
def OfferCraft(_=None):
    P = lclPlayer()
    if P.paused:
        return
    def Offer(space, _=None):
        sellprice = P.PlayerTrack.craftingTable.sell_value(space)
        persuasion = P.activateSkill("Persuasion")
        r = rbtwn(1, 4, None, sellprice+persuasion, 'Persuasion ')
        if r <= (sellprice + persuasion):
            def lvlUp(skill, _=None):
                P.updateSkill(skill, 1)
                exitActionLoop()()
            P.activateSkill("Persuasion")
            P.PlayerTrack.craftingTable.rmv_craft(space)
            output("The boy accepted your craft!", 'green')
            P.PlayerTrack.Quest.update_quest_status((1, 4), 'complete')
            if (P.skills['Crafting'] < 6) or (P.skills['Persuasion'] < 6):
                output("Choose a skill to level up:", 'blue')
                actions = {}
                if P.skills['Crafting'] < 6:
                    actions['Crafting'] = partial(lvlUp, 'Crafting')
                if P.skills['Persuasion'] < 6:
                    actions['Persuasion'] = partial(lvlUp, 'Persuasion')
                actionGrid(actions, False)
            else:
                output("Your Crafting and Persuasion are already level 6+", 'yellow')
                exitActionLoop()()
        else:
            output("Unable to convince the boy to accept your craft.", 'yellow')
            exitActionLoop()()
    actions = {'Cancel':exitActionLoop(amt=0)}
    if P.PlayerTrack.craftingTable.space_items[0] > 1:
        actions["Craft 1"] = partial(Offer, 0)
    if P.PlayerTrack.craftingTable.space_items[1] > 1:
        actions["Craft 2"] = partial(Offer, 1)
    actionGrid(actions, False)
    
def SpareWithBoy(_=None):
    P = lclPlayer()
    if P.paused:
        return
    attack = P.base[P.attributes["Attack"]]
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
    P.addItem('cooling cubes', -1)
    output("You applied the cubes successfully! The smith will now let you rent his facility for free!", 'green')
    P.free_smithing_rent = True
    P.PlayerTrack.Quest.update_quest_status((2, 1), 'complete')
    exitActionLoop()()

# Order: Action Name, Action Condition, Action Function
city_quest_actions = {(1, 1): ["Find Pet", "True", FindPet],
                      (1, 2): ["Clean House", "True", CleanHome],
                      (1, 3): ["Gaurd Home", "True", GaurdHome],
                      (1, 4): ["Gift Craft", "(self.playerTrack.craftingTable.space_items[0] <= 1) and (self.playerTrack.craftingTable.space_items[1] <= 1)", OfferCraft],
                      (1, 5): ["Spare with Boy", "True", SpareWithBoy],
                      (1, 6): ["Gift Book", "checkBook()", OfferBook],
                      (1, 7): ["Gift Sand", "'sand' in self.playerTrack.player.items", OfferSand],
                      (1, 8): ["Drop Baby Mammoth at Zoo", "hasattr(self.quests[(1, 8)], 'has_mammoth') and self.quests[(1, 8)].has_mammoth", ZooKeeper],
                      (2, 1): ["Apply Cubes", "'cooling cubes' in self.playerTrack.player.items", ApplyCubes]}
   
class Quest:
    def __init__(self, playerTrack):
        self.playerTrack = playerTrack
        qgrid = GridLayout(cols=1)
        self.Q = pd.read_csv('log\\QuestTable.csv')
        self.questDisplay = Button(text='', height=Window.size[1]*0.15,size_hint_y=None,color=(0,0,0,1),markup=True,background_color=(1,1,1,0))
        self.questDisplay.text_size = self.questDisplay.size
        self.questDisplay.bind(size=self.update_bkgSize)
        self.quests, data, self.stage_completion = {}, [], {i: 0 for i in range(1, 6)}
        for mission in range(1, 9):
            datarow = [Button(text=str(mission), disabled=True, background_disabled_normal='', color=(0,0,0,1))]
            for stage in range(1, 6):
                row = self.Q[(self.Q['Stage']==stage)&(self.Q['Mission']==mission)]
                msg = 'Requirement: ' + str(row['Requirements'].iloc[0]) + ' | Failable: ' + str(row['Failable'].iloc[0]) + ' | Reward: ' + str(row['Rewards'].iloc[0]) + '\nProcedure: '+ str(row['Procedure'].iloc[0])
                B = HoverButton(self.questDisplay, msg, text=row['Name'].iloc[0], disabled=False if (stage==1) and self.req_met((stage, mission)) else True)
                B.mission = mission
                B.stage = stage
                B.status = 'not started'
                datarow.append(B)
                self.quests[stage, mission] = B
                B.bind(on_press=self.activate)
            data.append(datarow)
        self.QuestTable = Table(['Mission','Stage 1\nCommon Folk', 'Stage 2\nNoblemen', 'Stage 3\nLocal Leaders', 'Stage 4\nCity Counsel', 'Stage 5\nMayor'], data, header_color=(50, 50, 50), header_as_buttons=True)
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
                if B.disabled: B.color = (1, 1, 1, 1)
    def req_met(self, quest):
        init_req = eval(quest_req[quest]) if quest in quest_req else True
        inCity = self.playerTrack.player.currenttile.tile == self.playerTrack.player.birthcity
        return init_req * inCity if (quest[0] == 1) else init_req * (self.stage_completion[quest[0]-1] >= 4) * inCity
    def update_quest_status(self, quest, new_status):
        if (new_status == 'started') and (self.quests[quest].status == 'not started'):
            self.quests[quest].disabled = True
            self.quests[quest].background_color = (1.5, 1.5, 0.5, 1.5)
            self.quests[quest].color = (0.6, 0.6, 0.6, 1)
        elif (new_status == 'not started') and (self.quests[quest].status != 'failed'):
            self.quests[quest].disabled = not self.req_met(quest)
            self.quests[quest].background_color = (1, 1, 1, 1)
            self.quests[quest].color = (1, 1, 1, 1)
        elif (new_status == 'failed') and (self.quests[quest].status == 'started'):
            output(f"You failed the mission '{self.quests[quest].text}'!", 'red')
            self.quests[quest].background_color = (3, 0, 0, 1.5)
        elif (new_status == 'complete') and (self.quests[quest].status == 'started'):
            output(f"You completed the mission '{self.quests[quest].text}'! Reputation increases by {self.quests[quest].stage}!", 'green')
            self.quests[quest].background_color = (0, 3, 0, 1.5)
            self.playerTrack.player.Reputation += self.quests[quest].stage
            self.playerTrack.reputationTab.text = f"Reputation: [color={self.playerTrack.hclr}]{self.playerTrack.player.Reputation}[/color]"
            self.stage_completion[self.quests[quest].stage] += 1
            for city in city_info:
                if self.playerTrack.player.Reputation > city_info[city]['entry']:
                    self.playerTrack.player.entry_allowed[city] = True
            # [INCOMPLETE] Get Reward -- probably similar to how we do req_met
        else:
            # Each statement was failed so assume that the status should not be changed
            return
        self.quests[quest].status = new_status
    def activate(self, instance):
        if self.playerTrack.player.paused:
            return
        quest = (instance.stage, instance.mission)
        if not self.req_met(quest):
            output("You do not meet the requirements for starting this quest!", 'yellow')
            return
        self.update_quest_status(quest, 'started')
        exitActionLoop('minor')()
    def add_active_city_actions(self, actions):
        for B in self.quests.values():
            if B.status == 'started':
                quest = (B.stage, B.mission)
                if (quest in city_quest_actions) and eval(city_quest_actions[quest][1]):
                    actions[city_quest_actions[quest][0]] = city_quest_actions[quest][2]
        return actions
        
class PlayerTrack(GridLayout):
    def __init__(self, player, **kwargs):
        super().__init__(**kwargs)
        self.player = player
        self.cols=1
        self.hclr = get_hexcolor((255, 85, 0))
        self.track_screen = ScreenManager()
        # Combat Screen
        combatGrid = GridLayout(cols=1)
        self.combatDisplay = Button(text='',height=Window.size[1]*0.05,size_hint_y=None,color=(0,0,0,1),markup=True,background_color=(1,1,1,0))
        self.Combat = Table(header=['Attribute','Base','Boost','Current'], data=self.get_Combat(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
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
        self.Knowledge = Table(header=['Skill','Level', 'XP'], data=self.get_Knowledge(), text_color=(0, 0, 0, 1), header_color=(50, 50, 50), header_as_buttons=True, color_odd_rows=True)
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
            #print(tabButton.color)
        self.tabs = GridLayout(cols=5, height=0.07*Window.size[1], size_hint_y=None)
        self.tab_color = (0.1, 0.6, 0.5, 1)
        self.combatTab = Button(text=f"Combat: [color={self.hclr}]{player.Combat}[/color]",markup=True,background_color=self.tab_color)
        self.combatTab.bind(on_press=partial(changeTab, 'Combat'))
        self.knowledgeTab = Button(text=f'Knowledge: [color={self.hclr}]{player.Knowledge}[/color]',markup=True,background_color=self.tab_color)
        self.knowledgeTab.bind(on_press=partial(changeTab, 'Knowledge'))
        self.capitalTab = Button(text=f'Capital: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.capitalTab.bind(on_press=partial(changeTab, "Capital"))
        self.reputationTab = Button(text=f'Reputation: [color={self.hclr}]0[/color]',markup=True,background_color=self.tab_color)
        self.reputationTab.bind(on_press=partial(changeTab, "Reputation"))
        self.itemsTab = Button(text='Items: 0/3',markup=True,background_color=self.tab_color)
        self.itemsTab.bind(on_press=partial(changeTab, 'Items'))
        self.tabs.add_widget(self.combatTab)
        self.tabs.add_widget(self.knowledgeTab)
        self.tabs.add_widget(self.capitalTab)
        self.tabs.add_widget(self.reputationTab)
        self.tabs.add_widget(self.itemsTab)
        # Add widgets in order
        self.add_widget(self.tabs)
        self.add_widget(self.track_screen)
    def get_Combat(self):
        data = []
        for i in range(len(self.player.atrorder)):
            atr = self.player.atrorder[i]
            if atr in {'Attack', 'Technique'}:
                adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in adept_loc[self.player.combatstyle]])
                mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in master_loc[self.player.combatstyle]])
            else:
                adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in adept_loc[atr]])
                mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in master_loc[atr]])
            if (atr[:4] == 'Def-') or (atr == 'Attack'):
                cmbtstyle = atr[4:] if atr[:4] == 'Def-' else self.player.combatstyle
                boosts = '\n[color=d66000]Boosts[/color]: '+', '.join([f'{i+1} - {inverse_ore_properties[cmbtstyle][i]}' for i in range(len(inverse_ore_properties[cmbtstyle]))])
            else:
                boosts = ''
            msg = adpt+' | '+mstr+boosts
            data.append([{'text':atr,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1),'hover':(self.combatDisplay, msg)},
                         {'text':str(self.player.combat[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.boosts[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.current[i]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}])
        return data
    def get_Knowledge(self):
        data = []
        for skill in self.player.skills:
            adpt = '[color=9f00ff]Adept Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in adept_loc[skill]])
            mstr = '[color=0041d6]Master Trainers[/color]: '+', '.join([city[0].upper()+city[1:] for city in master_loc[skill]])
            msg = adpt+' | '+mstr
            data.append([{'text':skill,'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1),'hover':(self.knowledgeDisplay, msg)},
                         {'text':str(self.player.skills[skill]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         {'text':str(self.player.xps[skill]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)}])
        return data
    def skirmDisplay(self, city, _=None):
        data = {'text':city[0].upper()+city[1:],'color':(0,0,0,1), 'disabled':True}
        fighting = set()
        for sk in Skirmishes[0]:
            if city in sk:
                fighting = fighting.union(sk.difference({city}))
        msg = '[color=009e33]Not at war![/color]' if len(fighting) == 0 else '[color=b50400]In skirmish with[/color]: '+', '.join([city[0].upper()+city[1:] for city in fighting])
        reputation = "\n" if self.player.entry_allowed[city] else f"\nEntry Reputation: [color={self.hclr}]{city_info[city]['entry']}[/color] | "
        msg += f"{reputation}Items Sold: "+', '.join(sorted(city_info[city]['sell']))
        data['hover'] = (self.capitalDisplay, msg)
        data['background_color'] = (1,1,1,0) if len(fighting) == 0 else (1, 0, 0, 0.3)
        return data
    def homeDisplay(self, city, _=None):
        data = {}
        msg = f' Capital: [color={self.hclr}]{capital_info[city]["home_cap"]}[/color], Capacity: {capital_info[city]["capacity"]}'
        if self.player.homes[city]:
            msg = 'Purchased! '+ msg
            data['disabled'] = True
            data['text'] = 'Owned!'
            data['background_color'] = (1, 10, 1, 0.8)
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
            else:
                msg = 'Owned!' + msg
                data['text'] = 'Automate'
                if self.player.currenttile.tile != city:
                    data['disabled'] = True
                else:
                    data['background_color'] = (2, 2, 0.2, 1)
                    data['func'] = partial(self.player.get_worker, city)
        else:
            msg = f'Cost: [color=cfb53b]{capital_info[city]["market"]}[/color]' + msg
            data['text'] = 'Buy'
            if self.player.currenttile.tile != city:
                data['disabled'] = True
            else:
                data['background_color'] = (2, 2, 0.2, 1)
                data['func'] = partial(self.player.purchase, 'market', city, None)
        data['hover'] = (self.capitalDisplay, city[0].upper()+city[1:]+' Market | '+msg)
        return data
    def villageDisplay(self, city, _=None):
        data = {}
        msg = f'Capital: [color={self.hclr}]1ea[/color], '+', '.join([v[0].upper()+v[1:]+': '+('[color=b50400]Not Invested[/color]' if v not in self.player.villages[city] else f'[color=bf004b]{self.player.villages[city][v][0]}[/color] ({self.player.villages[city][v][1]})') for v in city_villages[city]])
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
            else:
                sellprice = sellPrice[price] + self.player.bartering_mode
            clr = 'ffff75' if self.player.bartering_mode == 0 else '00ff75'
            cityset = self.player.currenttile.neighbortiles.intersection(set(cities))
            # Sell/Invest button:
            if ((self.player.currenttile.trader_rounds>0) or (self.player.currenttile.tile in cities)) and (item not in wares):
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
                        actionGrid({'Eat': partial(self.player.eat, item), 'Heat': partial(heatitem, item)}, False)
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
                    actionGrid({'Heat':partial(heatitem, item), 'Craft':partial(self.player.PlayerTrack.craftingTable.add_craft, item, None)}, False)
                usebutton = {'text':'Craft/Heat', 'func':craft_or_heat}
            elif (categ == 'Crafting') and (item not in noncraftableItems):
                usebutton = {'text':'Craft', 'func':partial(self.player.PlayerTrack.craftingTable.add_craft, item, None)}
            elif (categ == 'Smithing') and (self.player.currenttile.tile in cities):
                usebutton = {'text':'Smith', 'func':partial(self.player.PlayerTrack.armoryTable.add_slot, item, None, None, None)}
            else:
                usebutton = {'text':'','background_color':(1,1,1,0),'disabled':True}
            data.append([{'text':item, 'background_color':(1,1,1,0), 'disabled':True,'color':(0,0,0,1)},
                         {'text':str(self.player.items[item]),'disabled':True,'background_color':(1,1,1,0),'color':(0,0,0,1)},
                         sellbutton,
                         usebutton])
            # [INCOMPLETE] Add an "invest" button option if player on a village
        return data
    def updateAll(self):
        self.Combat.update_data_cells(self.get_Combat(), False)
        self.combatTab.text=f"Combat: [color={self.hclr}]{self.player.Combat}[/color]"
        self.Knowledge.update_data_cells(self.get_Knowledge(), False)
        self.knowledgeTab.text=f'Knowledge: [color={self.hclr}]{self.player.Knowledge}[/color]'
        self.Capital.update_data_cells(self.get_Capital())
        self.capitalTab.text=f"Capital: [color={self.hclr}]{self.player.Capital}[/color]"
        self.Items.update_data_cells(self.get_Items())
        self.itemsTab.text=f'Items: {self.player.item_count}/{self.player.max_capacity}'
        

cities = {'anafola':{'Combat Style':'Wizard','Coins':3,'Knowledges':[('Excavating',1),('Persuasion',1)],'Combat Boosts':[('Stability',2)]},
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
city_villages = {city: [] for city in cities} # Updated once the neighbors are defined
sellPrice = {1:0, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:5, 9:6}
mrktPrice = {1:1, 2:3, 3:5, 4:6, 5:8, 6:9, 7:10, 8:12, 9:20}
price2smithlvl = {1:1, 3:2, 6:3, 9:4}
gameItems = {'Food':{'raw meat':1,'cooked meat':2,'well cooked meat':3,'raw fish':1,'cooked fish':2,'well cooked fish':3,'fruit':2},
             'Crafting':{'string':1,'beads':1,'hide':1,'sand':1,'clay':2,'scales':2,'leather':2,'bark':2,'ceramic':3,'glass':5,'rubber':6,'gems':8},
             'Smithing':{'lead':1,'tin':1,'copper':1,'iron':1,'tantalum':3,'aluminum':3,'kevlium':3,'nickel':3,'tungsten':6,'titanium':6,'diamond':6,'chromium':6,'shinopsis':9,'ebony':9,'astatine':9,'promethium':9},
             'Knowledge Books':{'critical thinking book':8,'bartering book':5,'persuasion book':3,'crafting book':4, 'heating book':4,'smithing book':4,'stealth book':9,'survival book':6,'gathering book':5,'excavating book':8},
             'Cloth':{'benfriege cloth':4,'enfeir cloth':2,'glaser cloth':5,'pafiz cloth':2,'scetcher cloth':2,'starfex cloth':2,'tutalu cloth':2,'zinzibar cloth':2,'old fodker cloth':6,'old enfeir cloth':6,'old zinzibar cloth':6,'luxurious cloth':3},
             'Quests':{'cooling cubes':2}}
clothSpecials = {'benfriege cloth':{'zinzibar','glaser','enfeir','starfex'},
                 'enfeir cloth':{'benfriege','tutalu','pafiz'},
                 'glaser cloth':{'pafiz','fodker','benfriege','tutalu','scetcher'},
                 'pafiz cloth':{'zinzibar','glaser'},
                 'scetcher cloth':{'glaser'},
                 'starfex cloth':{'benfriege','tutalu','pafiz'},
                 'tutalu cloth':{'zinzibar','glaser','enfeir'},
                 'zinzibar cloth':{'benfriege','tutalu','pafiz','fodker'},
                 'old fodker cloth':{'zinzibar','glaser'},
                 'old zinzibar cloth':{'benfriege','tutalu','pafiz','fodker'},
                 'luxurious cloth':{}}
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
        
def city_trainer(abilities, mastery, _=None):
    actions = {ability:partial(Train, ability, mastery, False) for ability in abilities}
    actions["Back"] = exitActionLoop(amt=0)
    actionGrid(actions, False)

def city_actions(city, _=None):
    P = lclPlayer()
    def Sparring(_=None):
        if not P.paused:
            encounter('Sparring Partner', [2, 5], [cities[city]['Combat Style']], {}, enc=0)
    def Duelling(_=None):
        if (not P.paused) and (P.dueling_hiatus==0):
            P.dueling_hiatus = 2
            encounter('Duelist', [max([2, sum(P.current)-2]), sum(P.current)+6], ['Physical', 'Elemental', 'Trooper', 'Wizard'], {'coins':int(P.Combat/10)+2}, enc=0)
    def Inn(_=None):
        if P.paused:
            return
        if P.coins == 0:
            output("Insufficient coin", 'yellow')
            return
        P.coins -= 1
        P.recover(2)
    T = game_app.game_page.board_page.citytiles[city]
    actions = {'Market':partial(Trading, False),
               'Adept':partial(city_trainer, T.adept_trainers, 'adept'),
               'Master':partial(city_trainer, T.master_trainers, 'city')}
    if (P is not None) and (P.currenttile.tile == 'scetcher'):
        actions['Duel'] = Duelling
    elif (P is not None) and (P.birthcity == city): 
        actions['Sparring'] = Sparring
    if (P is not None) and (P.birthcity != city) and (not P.homes[city]):
        actions['Inn (2): 1 coin'] = Inn
    if P.currenttile.tile == P.birthcity:
        # If player is in their birth city then allow more actions depending on active quests
        actions = P.PlayerTrack.Quest.add_active_city_actions(actions)
    actionGrid(actions, True)

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
        self.adept_trainers = set()
        self.master_trainers = set()
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
        if self.tile in cities:
            city_villages[self.tile] = sorted(self.neighbortiles.intersection({'village1','village2','village3','village4','village5'}))
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
                categ, price = getItemInfo(item)
                L.append(f'{item}: [color=ffff75]{gameItems[categ][item]}[/color]')
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
        if self.parentBoard.inspect_mode:
            if self.tile in avail_actions:
                self.parentBoard.game_page.inspectTile(*avail_actions[self.tile](inspect=True))
        else:
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
        self.inspect_mode = False
        self.gridtiles = {}
        self.citytiles = {}
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
            if T.tile in cities:
                self.citytiles[T.tile] = T
                # Initial markets will be different for local players
                T.city_wares = set(np.random.choice(list(city_info[T.tile]['sell']), 6))
            T.set_neighbors()
        self.zoomButton = Button(text="Zoom", pos_hint={'x':0,'y':0}, size_hint=(0.06, 0.03))
        self.zoomButton.bind(on_press=self.updateView)
        self.add_widget(self.zoomButton)
        self.inspectButton = Button(text='Inspect', pos_hint={'x':0.07,'y':0}, size_hint=(0.06, 0.03))
        self.inspectButton.bind(on_press=self.toggleInspect)
        self.add_widget(self.inspectButton)
    def add_tile(self, tiletype, x, y):
        T = Tile(tiletype, x, y)
        self.gridtiles[(x,y)] = T
        T.parentBoard = self
        self.add_widget(T)
    def startRound(self):
        for P in self.Players.values():
            P.paused = False
        if self.localPlayer.dueling_hiatus > 0:
            self.localPlayer.dueling_hiatus -= 1
        self.localPlayer.unsellable = set()
        self.localPlayer.actions = self.localPlayer.max_actions
        self.sendFaceMessage('Start Round!')
        for T in self.gridtiles.values():
            T.update_tile_properties()
        self.localPlayer.receiveInvestments() # Receive village investments (or at least update)
        self.localPlayer.update_mainStatPage()
        paralyzed = check_paralysis()
        if not paralyzed: 
            self.localPlayer.started_round = True
            self.localPlayer.go2consequence(0)
    def update_market(self, city_markets, _=None):
        for city, T in self.citytiles.items():
            T.city_wares = city_markets[city]
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
            #self.game_page.eat_button.bind(on_press=self.localPlayer.eat())
            self.localPlayer.add_mainStatPage()
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
            
            # Begin city action loop
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
    def sendFaceMessage(self, msg=None, clr=None, scheduleTime=2):
        timeNow = time()
        msg = f'[color={trns_clr[clr]}]{msg}[/color]' if clr is not None else msg
        if msg is not None:
            self.startLblMsgs.append(msg)
            self.startLblTimes.append(timeNow)
        max_i = -1
        for i in range(len(self.startLblTimes)):
            if (timeNow - self.startLblTimes[i]) > scheduleTime:
                max_i = i
        self.startLblMsgs = self.startLblMsgs[(max_i+1):]
        self.startLblTimes = self.startLblTimes[(max_i+1):]
        self.startLabel.text = '\n'.join(self.startLblMsgs)
        if msg is not None: 
            if msg == 'Start Round!': msg = '\n'+msg
            output(msg, clr)
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
    def toggleInspect(self, _=None):
        self.inspect_mode = not self.inspect_mode
        self.inspectButton.text = 'Move' if self.inspect_mode else 'Inspect'
        self.game_page.outputscreen.current = 'Inspect' if self.inspect_mode else 'Actions'

class ScrollLabel(Label, HoverBehavior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.msg_index = 0
        self.messages = []
        self.hovering = False
    def add_msg(self, msg):
        self.msg_index += 1
        self.messages.append(msg)
        self.display()
    def display(self):
        self.text = '\n'.join(self.messages[:self.msg_index])
    def on_enter(self):
        self.hovering = True
    def on_leave(self):
        self.hovering = False
    def on_touch_down(self, touch):
        if self.hovering and touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                self.msg_index = max([0, self.msg_index - 1])
            elif touch.button == 'scrollup':
                self.msg_index = min([len(self.messages), self.msg_index + 1])
            self.display()

class GamePage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.board_page = BoardPage(self)
        screen = Screen(name='Board')
        screen.add_widget(self.board_page)
        self.main_screen.add_widget(screen)
        self.add_widget(self.main_screen)
        # Add the Side Screen
        input_y, label_y, stat_y, action_y, output_y, toggle_y = 0.05, 0.25, 0.1, 0.2, 0.35, 0.05
        self.stat_ypos, self.stat_ysize = input_y+label_y, stat_y
        self.right_line = RelativeLayout(size_hint_x=self.right_line_x)
        self.toggleView = Button(text="Player Track",pos_hint={'x':0,'y':(input_y+label_y+stat_y+action_y+output_y)},size_hint=(1,toggle_y),background_color=(0, 0.4, 0.4, 1))
        self.toggleView.bind(on_press=self.switchView)
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
        self.actionGrid = GridLayout(pos_hint={'x':0,'y':(input_y+label_y+stat_y)},size_hint_y=action_y,cols=2)
        self.recover_button = Button(text='Rest (2)')
        # Button is bound after local player is detected
        self.actionButtons = [self.recover_button]
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
        self.right_line.add_widget(self.toggleView)
        self.right_line.add_widget(self.outputscreen)
        self.right_line.add_widget(self.actionGrid)
        self.right_line.add_widget(self.display_page)
        self.right_line.add_widget(self.new_message)
        self.add_widget(self.right_line)
        # Any keyboard press will trigger the event:
        Window.bind(on_key_down=self.on_key_down)
    def switchView(self, instance):
        if self.main_screen.current != 'Battle':
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
            self.actionButtons = [self.recover_button]#, self.eat_button]
            self.actionGrid.add_widget(self.recover_button)
            #self.actionGrid.add_widget(self.eat_button)
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
        #self.output.text += '\n' + message
        self.output.add_msg(message)
    def update_display(self, username, message):
        if message[0]=='\n':
            message = message[1:]
        clr = get_hexcolor((131,215,190)) if username == self.board_page.localuser else get_hexcolor((211, 131, 131))
        #self.display_page.text += f'\n|[color={clr}]{username}[/color]| ' + message
        self.display_page.add_msg(f'\n|[color={clr}]{username}[/color]| ' + message)
    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        # We want to take an action only when Enter key is being pressed, and send a message
        if keycode == 40:
            # Send Message
            message = self.new_message.text
            self.new_message.text = ''
            if message:
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
        
def notAtWar(skrm):
    s = set()
    for S in skrm:
        for city in S:
            s.add(city)
    return set(cities).difference(s)

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
        elif category == '[SKIRMISH]':
            Skirmishes[0] = message
            Clock.schedule_once(partial(game_app.game_page.board_page.localPlayer.getIncome), 0.2)
        elif category == '[MARKET]':
            Clock.schedule_once(partial(game_app.game_page.board_page.update_market, message), 0.02)
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