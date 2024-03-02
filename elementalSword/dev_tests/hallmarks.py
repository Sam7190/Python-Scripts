# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:12:39 2024

@author: samir
"""

from hallmark_variables import hallmarks as hvb

import numpy as np
from time import time
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

class Test_Player():
    def __init__(self):
        
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
        self.player_fight_hiatus = {}
        self.free_smithing_rent = False
        self.city_discount = 0
        self.city_discount_threshold = float('inf')
        self.standard_read_xp = 4
        self.group = {}
        self.has_warrior = 0
        self.training_discount = False
        self.round_num = 0
        
        self.coins = 0
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
        self.current = self.combat + self.boosts
        #Knowledge
        self.skills = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
        self.xps = {'Critical Thinking':0, 'Bartering':0, 'Persuasion':0, 'Crafting':0, 'Heating':0, 'Smithing':0, 'Stealth':0, 'Survival':0, 'Gathering':0, 'Excavating':0}
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
        self.grand_bank = {'credit': 0, 'borrow_rounds': None, 'strikes': 0} # demetry grand central bank
        self.ninja_lair = {'sharpness': 0, 'invincibility': 0, 'vanish': 0, 'speed': 0, 'vision': 0} # zinzibar ninja lair
        self.meditation = {'class': 1, 'score': 0, 'success': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}} # pafiz meditation chamber
        self.creature = {'delay': 0, 'lvl': 0, 'rounds': 0, 'bond': 0, 'combat_limit': 1, 'combats': 0}  # anafola castle of conjurors
        self.reaquisition = {'class': 1, 'success': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}} # enfeir requisition guild
        self.wizard_membership = None # Tamariza wizard tower
        self.ancestrial_order = {'loot': 0, 'food': 0, 'fatigue': 0, 'minor': False, 'horse': False, 'major': False} # kubani ancestrial order
        self.gifted_museum = 0 # starfex ancient magic museum
        # Titles
        self.titles = {'explorer': {'titleVP': 5, 'minTitleReq': 20, 'value':0, 'category': 'General', 'description': 'Most unique tiles traveled upon.'},
                      'loyal': {'titleVP': 2, 'minTitleReq': 25, 'value':0, 'category': 'General', 'in_birthcity':True, 'description': 'Most rounds spent in their birth city (all actions per round).'},
                      'valiant': {'titleVP': 3, 'minTitleReq': 6, 'value':0, 'category': 'Combat', 'description': 'Maximum difference between an opponent stronger than you, that you defeated, and your total combat at the start of battle.'},
                      'sorcerer': {'titleVP': 5, 'minTitleReq': 2, 'value':0, 'category': 'Fellowship', 'description': 'Longest continuous Tamariza Wizard Tower premium holder.'},
                      'superprime': {'titleVP': 5, 'minTitleReq': 40, 'value':0, 'category': 'Fellowship', 'description': 'Highest credit score from the Demetry Grand Bank'},
                      'traveler': {'titleVP': 3, 'minTitleReq': 30, 'value':0, 'category': 'General', 'description': 'Most road tile movement.'},
                      'apprentice': {'titleVP': 4, 'minTitleReq': 60, 'value':0, 'category': 'Knowledge', 'description': 'Most cumulative levels gained from trainers.'},
                      'scholar': {'titleVP': 5, 'minTitleReq': 5, 'value':0, 'category': 'Fellowship', 'description': 'Most books checked out from the Benefriege public library, and learned from.'},
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


class benfriege(FloatLayout):
    def __init__(self, player, **kwargs):
        super().__init__(**kwargs)
        self.player = player
        self.hallmark = hvb['benfriege']['hallmark']
        self.description = hvb['benfriege']['description']
        