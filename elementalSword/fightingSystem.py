# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:38:37 2024

@author: samir
"""


from kivy.uix.floatlayout import FloatLayout

from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color,Rectangle
from kivy.clock import Clock
from kivy.core.window import Window

import numpy as np
import essentialfuncs as essf

#%% Fighting System

class HitBox(Button):
    def __init__(self, fpage, **kwargs):
        super().__init__(**kwargs)
        eval('logging').debug("Making hitbox")
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
        eval('Clocked')(self.endAttack, self.TimeRemaining, 'End Attack')
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
                eval('Clocked')(self.npcDefends, 1, "NPC Defending")
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
            dis2cent =essf.euc(nrmPos, np.array([0.5, 0.5]))
            if vgID > 0:
                for gi in group2boxes[vgID-1]:
                    if not self.fakes[gi]:
                        dis2last =essf.euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
                        break
            else:
                dis2last = dis2cent
            total, dis2group = 0, 0
            for gi in group2boxes[vgID]:
                if gi == i:
                    continue
                total += 1
                dis2group +=essf.euc(nrmPos, self.getNormalizedPosition(self.boxes[gi].pos))
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
                eval('logging').info(f"Click Probability {click_p} {self.fakes[i]}")
                clickType = None
                if np.random.rand() <= click_p:
                    if not self.fakes[i]: clicked_fake = False
                    # This means the AI "clicked" the box - now check to see if they would have "dodged" the box
                    eval('logging').info(f"Clicked! Dodge Probability {dodge_p} {self.fakes[i]}")
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
                eval('logging').info(f"Click Type: {clickType}")
                self.npcClickBox(i, clickType, 0.3 + 0.3*vgID, 0.3)
            for i in boxIDs:
                # Any remaining ids must be fake, so remove them
                self.npcClickBox(i, None, 0.3 + 0.3*vgID, 0.3)
            #eval('Clocked')(partial(self.removeBox, i, damageNPC), 0.3 + 0.3*vgID)
        try:
            vgID
        except NameError:
            vgID = 0
        eval('Clocked')(self.fpage.nextAttack, 0.9 + 0.3*vgID, "Go to Next Attack")
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
            eval('Clocked')(partial(self.removeBox, i, dmg), rmvTime, "Remove hit box")
        eval('Clocked')(clk, delay, "NPC click box delay")
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
        d2c =essf.euc(npos, np.array([0.5, 0.5]))
        if vg > 0:
            for gi in self.volley_group[vg-1]:
                if not self.live_rects[gi]['fake']:
                    d2p =essf.euc(npos, self.live_rects[gi]['nrmpos'])
                    break
        else:
            d2p = d2c
        d2g, total = 0, 0
        for gi in self.volley_group[vg]:
            if gi == i:
                continue
            d2g +=essf.euc(npos, self.live_rects[gi]['nrmpos'])
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
        eval('Clocked')(transition, self.dodgeTime if prevState=='dodge' else self.blockTime, 'block/dodge time')
    def removeBox(self, i, _=None):
        def rmvBox(_=None):
            self.live_rects[i]['removed'] = True
            self.fpage.canvas.after.remove(self.live_rects[i]['rect'])
            if self.fpage.logDef:
                self.add2Log(i)
        if not self.live_rects[i]['removed']:
            eval('Clocked')(rmvBox, 0.1, 'remove box')
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
        
class SkillPage():
    def __init__(self, skill):
        eval('logging').debug(f"Initiating SkillPage for {skill}")
        
        # Player objects
        self.P = lclPlayer()
        self.P.paused = True
        

class FightPage(FloatLayout):
    def __init__(self, name, style, lvl, stats, encountered=True, logDef=False, reward=None, consume=None, action_amt=1, foeisNPC=True, background_img=None, consequence=None, foeStealth=0, **kwargs):
        super().__init__(**kwargs)
        eval('logging').debug("Initiatating Fight Page")
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
        eval('logging').info(f"Player Stats: {self.pstats}")
        eval('logging').info(f"Foe Stats: {self.foestats}")
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
                eval('Clocked')(self.triggerNext, delay, 'trigger next attack')
        eval('Clocked')(trigger, delay, 'trigger attack trigger')
    def triggerNext(self, _=None):
        if self.fightorder[self.order_idx] == self.P.username:
            self.playerAttacks()
        else:
            self.playerDefends()
    def playerAttacks(self):
        eval('logging').debug("Player is attacking")
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
            eval('logging').debug("Scaling hit box")
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
        eval('logging').info(f"Foe is taking damage. Prior HP - {self.foestats[self.P.attributes['Hit Points']]}")
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
        eval('logging').info(f"Player is taking damage. Prior HP - {self.pstats[self.P.attributes['Hit Points']]}")
        if (self.pstats[self.P.attributes['Hit Points']] > 0) and (self.fighting):
            self.pstats[self.P.attributes['Hit Points']] = max([0, self.pstats[self.P.attributes['Hit Points']]-amt])
            cell = self.statTable.cells[self.P.username][self.P.attributes['Hit Points']]
            cell.text = str(self.pstats[self.P.attributes['Hit Points']])
            cell.background_color = (1, 1, 0.2, 0.6)
            if self.pstats[self.P.attributes['Hit Points']] == 0:
                if self.foename == 'Duelist':
                    self.P.dueling_hiatus = 3 # Losing makes it harder to go back into arena
                self.endFight()