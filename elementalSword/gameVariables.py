# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:43:33 2024

@author: samir
"""
import numpy as np

#%% Game Properties
food_restore = {'raw meat':(1,0),'cooked meat':(2,0),'well cooked meat':(3,0),
                'raw fish':(0,1),'cooked fish':(0,2),'well cooked fish':(0,3),
                'fruit':(1,1)}

ore_properties = {'lead':('Elemental',1),'tin':('Physical',1),'copper':('Trooper',1),'iron':('Wizard',1),
                  'tantalum':('Elemental',2),'aluminum':('Physical',2),'kevlium':('Trooper',2),'nickel':('Wizard',2),
                  'tungsten':('Elemental',3),'titanium':('Physical',3),'diamond':('Trooper',3),'chromium':('Wizard',3),
                  'shinopsis':('Elemental',4),'ebony':('Physical',4),'astatine':('Trooper',4),'promethium':('Wizard',4)}

capital_info = {'anafola':{'home':40,'home_cap':5,'capacity':4,'market':20,'return':5,'market_cap':2,'invest':8,'efficiency':0,'discount':0,'trader allowed':False},
                'benfriege':{'home':8,'home_cap':2,'capacity':4,'market':3,'return':1,'market_cap':1,'invest':3,'efficiency':0,'discount':0,'trader allowed':False},
                'demetry':{'home':49,'home_cap':6,'capacity':4,'market':24,'return':6,'market_cap':2,'invest':9,'efficiency':0,'discount':0,'trader allowed':False},
                'enfeir':{'home':20,'home_cap':4,'capacity':4,'market':9,'return':2,'market_cap':1,'invest':3,'efficiency':0,'discount':0,'trader allowed':False},
                'fodker':{'home':24,'home_cap':4,'capacity':4,'market':12,'return':3,'market_cap':1,'invest':None,'efficiency':0,'discount':0,'trader allowed':False}, # Fodker has no villages to invest in
                'glaser':{'home':5,'home_cap':2,'capacity':4,'market':3,'return':1,'market_cap':1,'invest':3,'efficiency':0,'discount':0,'trader allowed':False},
                'kubani':{'home':37,'home_cap':5,'capacity':4,'market':19,'return':4,'market_cap':2,'invest':7,'efficiency':0,'discount':0,'trader allowed':False},
                'pafiz':{'home':27,'home_cap':4,'capacity':4,'market':13,'return':3,'market_cap':1,'invest':5,'efficiency':0,'discount':0,'trader allowed':False},
                'scetcher':{'home':42,'home_cap':5,'capacity':4,'market':20,'return':5,'market_cap':2,'invest':8,'efficiency':0,'discount':0,'trader allowed':False},
                'starfex':{'home':28,'home_cap':4,'capacity':4,'market':14,'return':3,'market_cap':1,'invest':5,'efficiency':0,'discount':0,'trader allowed':False},
                'tamarania':{'home':43,'home_cap':5,'capacity':4,'market':21,'return':5,'market_cap':2,'invest':8,'efficiency':0,'discount':0,'trader allowed':False},
                'tamariza':{'home':42,'home_cap':5,'capacity':4,'market':21,'return':5,'market_cap':2,'invest':8,'efficiency':0,'discount':0,'trader allowed':False},
                'tutalu':{'home':23,'home_cap':4,'capacity':4,'market':10,'return':3,'market_cap':1,'invest':4,'efficiency':0,'discount':0,'trader allowed':False},
                'zinzibar':{'home':8,'home_cap':2,'capacity':4,'market':2,'return':1,'market_cap':1,'invest':3,'efficiency':0,'discount':0,'trader allowed':False}}
home_price_order = sorted(capital_info, key=lambda x: capital_info[x]['home'])

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
             'tutalu':{'Hit Points':8, 'Trooper':12, 'Def-Physical':8, 'Smithing':8, 'Excavating':8, 'entry':8, 'sell':{'raw meat', 'cooked meat', 'string', 'beads', 'hide', 'leather', 'copper', 'kevlium', 'diamond', 'excavating book'}},
             'zinzibar':{'Agility':12, 'Physical':8, 'Def-Wizard':8, 'Persuasion':8, 'Smithing':8, 'Stealth':12, 'entry':3, 'sell':{'raw meat', 'cooked meat', 'string', 'hide', 'lead', 'tin', 'tantalum', 'aluminum'}}}

skills = ['Critical Thinking', 'Bartering', 'Persuasion', 'Crafting', 'Heating', 'Smithing', 'Stealth', 'Survival', 'Gathering', 'Excavating']

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

#%% City and Board Properties
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

library_books = [f'{level} {skill}'.lower() for level in ['Beginner', 'Intermediate', 'Advanced'] for skill in skills]
lib_book_dict = {lb: None for lb in library_books}
lib_book_dict['learned library book'] = None

gameItems = {'Food':{'raw meat':1,'cooked meat':2,'well cooked meat':3,'raw fish':1,'cooked fish':2,'well cooked fish':3,'fruit':2},
             'Crafting':{'string':1,'beads':1,'hide':1,'sand':1,'clay':2,'scales':2,'leather':2,'bark':2,'ceramic':3,'glass':5,'rubber':6,'gems':8},
             'Smithing':{'lead':1,'tin':1,'copper':1,'iron':1,'tantalum':3,'aluminum':3,'kevlium':3,'nickel':3,'tungsten':6,'titanium':6,'diamond':6,'chromium':6,'shinopsis':9,'ebony':9,'astatine':9,'promethium':9},
             'Knowledge Books':{'critical thinking book':8,'bartering book':5,'persuasion book':3,'crafting book':4, 'heating book':4,'smithing book':4,'stealth book':9,'survival book':6,'gathering book':5,'excavating book':8},
             'Cloth':{'benfriege cloth':4,'enfeir cloth':2,'glaser cloth':5,'pafiz cloth':2,'scetcher cloth':2,'starfex cloth':2,'tutalu cloth':2,'zinzibar cloth':2,'old fodker cloth':6,'old enfeir cloth':6,'old zinzibar cloth':6,'luxurious cloth':3},
             'Quests':{'cooling cubes':2},
             # Hallmark Groups
             'GrandLibrary': lib_book_dict
             }
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
valid_items = []
for category in gameItems:
    valid_items += list(gameItems[category].keys())
valid_items += list(clothSpecials.keys())
valid_items = set(valid_items)

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

#%% City Page Properties
corner_gate_size = (0.04, 0.0175)
edge_gate_size = (0.022, 0.0395)
gate_color = (0.746, 0.302, 0.145, 0.4)

gates = {'sw': {'pos_hint': {'x': 0.045, 'y': 0.03}, 'size_hint': corner_gate_size, 'background_color': gate_color},
         'nw': {'pos_hint': {'x': 0.045, 'y': 0.947}, 'size_hint': corner_gate_size, 'background_color': gate_color},
         'se': {'pos_hint': {'x': 0.935, 'y': 0.03}, 'size_hint': corner_gate_size, 'background_color': gate_color},
         'ne': {'pos_hint': {'x': 0.935, 'y': 0.947}, 'size_hint': corner_gate_size, 'background_color': gate_color},
         'e':  {'pos_hint': {'x': 0.972, 'y': 0.523}, 'size_hint': edge_gate_size, 'background_color': gate_color},
         'w':  {'pos_hint': {'x': 0.005, 'y': 0.523}, 'size_hint': edge_gate_size, 'background_color': gate_color}}

shack_rest = {'pos_hint': {'x': 0.721, 'y': 0.326}, 'size_hint': (0.026, 0.026), 'background_color': (0.561, 0.337, 0.231, 0.4)}

# hex colors indicate corresponding selection image
region_colors = {'000000': None,
                 '736a37': 'shack',
                 'fb36a3': 'gates',
                 'cd1111': 'skirmishes',
                 '76428a': 'inn', # 'Inn (3): 1 coin'
                 '36fb9d': 'market', # 'Market'
                 'ce853b': 'sparring', # 'Sparring'
                 '3657fb': 'house',
                 '99e068': 'personal_market',
                 'fb5e36': 'jobs', # Job Posting
                 'b37ec8': 'smithing',
                 # People
                 '6a7508': 'mayor', # S5
                 '752e0c': 'district_counselor_1', # S4M5+7
                 '682a0c': 'district_counselor_2', # S4M4+6
                 '592d18': 'district_counselor_3', # S4M2+8
                 '49221b': 'district_counselor_4', # S4M1+3
                 'f8934d': 'smither', # S2M1+7
                 'fb365e': 'district_leader_1', # S3M1+5
                 '6abe30': 'district_leader_2', # S3M2+3
                 '97d8ba': 'district_leader_3', # S3M4+6
                 '59a524': 'district_leader_4', # S3M7+8
                 # Quest Lines
                 '92c96b': 'lost_pet', # S1M1
                 'b9c1c1': 'untidy_home', # S1M2
                 '666288': 'gaurd_duty', # S1M3
                 '36fb57': 'something_special', # S1M4
                 'b39190': 'young_warrior', # S1M5
                 '8f974a': "librarian's_son", # S1M6
                 'a3deff': 'play_pin', # S1M7
                 'e9d4d4': 'animal_keeper', # S1M8
                 'd38560': 'mystery_robber', # S2M2
                 '8d2b2f': 'protection_services', # S2M3
                 '07793d': 'mother_serpent', # S2M4
                 'ff1515': "librarian's_secret", # S2M5
                 '4f2902': 'the_letter', #S2M6
                 '5d6167': "ninja's_way", #S2M8
                # Trainers
                 '169f3c': 'excavating_trainer',
                 '68c481': 'criticalthinking_trainer',
                 '7af79d': 'cunning_trainer',
                 'c9c537': 'crafting_trainer',
                 '898629': 'persuasion_trainer',
                 'e2dd02': 'bartering_trainer',
                 'e22402': 'agility_trainer',
                 'c48377': 'stealth_trainer',
                 '7a3225': 'hitpoints_trainer',
                 '841400': 'stability_trainer',
                 'e7e21e': 'wizard_trainer',
                 '9f9b16': 'elemental_trainer',
                 'c4c164': 'physical_trainer',
                 'fcf622': 'trooper_trainer',
                 '424239': 'def-trooper_trainer',
                 '45452d': 'def-elemental_trainer',
                 '4a491f': 'def-physical_trainer',
                 '31311f': 'def-wizard_trainer',
                 '504c8e': 'smithing_trainer',
                 '48d3c1': 'survival_trainer',
                 '25ac9b': 'heating_trainer',
                 '82d6cc': 'gathering_trainer',
                # Hallmarks
                 '00a483': 'castle_of_conjurors', # anafola
                 '2cb89c': 'grand_library', # benfriege
                 'caf8ef': 'grand_bank', # demetry
                 '2f6e61': 'reaquisition_guild', # enfeir
                 '1f9f85': 'defenders_guild', # fodker
                 '98bdb6': 'peace_embassy', # glaser
                 '3f7996': 'ancestrial_order', # kubani
                 '85d8c9': 'meditation_chamber', # pafiz
                 '083044': 'colosseum', # scetcher
                 '395969': 'ancient_magic_museum', # starfex
                 'a7b2b8': 'smithing_guild', # tamarania
                 'c4ebff': 'wizards_tower', # tamariza
                 '195776': 'hunters_guild', # tutalu
                 '01141d': 'hidden_lair' # zinzibar
                 }

action_mapper = {'shack': 'Rest (2)',
                 'inn': 'Inn (3): 1 coin',
                 'market': 'Market',
                 'sparring': 'Sparring',
                 'jobs': 'Job Posting'}

hallmark_mapper = {'castle_of_conjurors': 'anafola',
                   'grand_library': 'benfriege',
                   'grand_bank': 'demetry',
                   'reaquisition_guild': 'enfeir',
                   'defenders_guild': 'fodker',
                   'peace_embassy': 'glaser',
                   'ancestrial_order': 'kubani',
                   'meditation_chamber': 'pafiz',
                   'colosseum': 'scetcher',
                   'ancient_magic_museum': 'starfex',
                   'smithing_guild': 'tamarania',
                   'wizards_tower': 'tamariza',
                   'hunters_guild': 'tutalu',
                   'hidden_lair': 'zinzibar'}

inverse_region_colors = {value: key for key, value in region_colors.items()}

region_quest_mapper = {'mayor': [(5,1), (5,2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)],
                 'district_counselor_1': [(4, 5), (4, 7)], # S4M5+7
                 'district_counselor_2': [(4, 4), (4, 6)], # S4M4+6
                 'district_counselor_3': [(4, 2), (4, 8)], # S4M2+8
                 'district_counselor_4': [(4, 1), (4, 3)], # S4M1+3
                 'smither': [(2, 1), (2, 7)], # S2M1+7
                 'district_leader_1': [(3, 1), (3, 5)], # S3M1+5
                 'district_leader_2': [(3, 2), (3, 3)], # S3M2+3
                 'district_leader_3': [(3, 4), (3, 6)], # S3M4+6
                 'district_leader_4': [(3, 7), (3, 8)], # S3M7+8
                 # Quest Lines
                 'lost_pet': [(1, 1)], # S1M1
                 'untidy_home': [(1, 2)], # S1M2
                 'gaurd_duty': [(1, 3)], # S1M3
                 'something_special': [(1, 4)], # S1M4
                 'young_warrior': [(1, 5)], # S1M5
                 "librarian's_son": [(1, 6)], # S1M6
                 'play_pin': [(1, 7)], # S1M7
                 'animal_keeper': [(1, 8)], # S1M8
                 'mystery_robber': [(2, 2)], # S2M2
                 'protection_services': [(2, 3)], # S2M3
                 'mother_serpent': [(2, 4)], # S2M4
                 "librarian's_secret": [(2, 5)], # S2M5
                 'the_letter': [(2, 6)], #S2M6
                 "ninja's_way": [(2, 8)] #S2M8
                       }
#quest_button_size = (0.05, 0.02)
multi_button_size = (0.04, 0.04)

inverse_quest_mapper = {}
for person, quests in region_quest_mapper.items():
    for quest in quests:
        inverse_quest_mapper[quest] = person

#%% Mappers
action_color_map = {'*b': {'text': (0, 0, 0, 1), 'background': (0.192, 0.737, 1.0, 1.0)},
                    '*g': {'text': (0, 0, 0, 1), 'background': (0.192, 0.929, 0.192, 1.0)}}

#%% Hallmarks
hallmarks = {
    'benfriege': 
        {'hallmark': 'Grand Library',
         'attribute': 'grand_library',
         'description': 
"""At this library you can read books or check them out from a choice of six random per round, incurring a minor action.

  - Read Books: Gain fatigue equal to your reading fatigue. You can clear this fatigue by not reading for 
                two consecutive major actions, including knowledge books.
        > Reading fatigue increases by one
        > Beginner (lvl 0-3): 1xp - Success between Critical Thinking 0-4
        > Intermediate (lvl 4-5): 1xp - Success between Critical Thinking 1-8
        > Advanced (lvl 6-7): 1xp - Success between Critical Thinking 2-12
                   
  - Checkout Books: A trainer must teach you the book. Following checkout go to an adept or master trainer of that skill to learn.
        > Beginner (lvl 0-3): 3xp for 2 coin
        > Intermediate (lvl 4-5): 4xp for 3 coins
        > Advanced (lvl 6-7): 4xp for 4 coins"""
        },
    'demetry':
        {'hallmark': 'Grand Bank',
         'attribute': 'grand_bank',
         'description':
"""This bank provides no interest loans.

  - Maximum amount of Loan: Credit Score + (Reputation / 2) + 1
  
  - Length of Loan: 10 rounds + persuasion skill.
  
  - If the money is returned between 5 rounds and the original length of the loan:
        > Credit Score increases by the loan amount.
  
  - You must be present at Demetry to return the loan.
  
  - It is possible to extend the loan by 5 rounds after the end of term:
        > If your credit score is zero then you are given no redemption period (skip strikes below).
        > 1st Strike: (persuasion / 12) chance of no impact on credit score, otherwise 25% cut.
        > 2nd Strike: (persuasion / 18) chance of no impact on credit score, otherwise 75% cut.
  
  - Consequence for failing to repay the bank after all chances given:
        > You can no longer bank at the Grand Bank and credit score goes to -1.
        > Your coins are automatically deducted to match the loan amount.
        > If you do not have enough coins, your homes are sold to the bank from least expensive to most until nothing is owed.
        > If you still owe the bank money, then the rest of the owed amount is forgiven."""
        },
    'kubani':
        {'hallmark': 'Ancestrial Order',
         'attribute': 'ancestrial_order',
         'description':
"""An ancient order that studies and teaches the arts of interacting with your environment, and are devoted to pass down ancient traditions.

There are three primary ancient knowledges: 1) Loot, 2) Food, and 3) Fatigue.
  > Each have 3 classes to progress through, completing the final class gives 2 VP.
There are also three exclusive ancient knowledges: 1) Minor Treading [Class 1], 2) Horse Riding [Class 2], and 3) Major Treading [Class 3].

Learning Requirements:
    - Kubani residents subtract 1 skill knowledge requirement or 5 total knowledge requirement.
    
    * [Loot] Class 1: Lvl 2 Gathering | Class 2: Lvl 6 Gathering | Class 3: Lvl 10 Gathering
    * [Food] Class 1: Lvl 2 Heating | Class 2: Lvl 6 Heating | Class 3: Lvl 10 Heating 
    * [Fatigue] Class 1: Lvl 2 Survival | Class 2: Lvl 6 Survival | Class 3: Lvl 10 Survival
    * Minor Treading (class 1): 15 Total Knowledge | Horse Riding (class 2): 45 Total Knowledge | Major Treading (class 3): 80 Total Knowledge

Session Required to Progress (Major Action):
    * Class 1: 2 successful sessions; chance of success is Critical Thinking / 8
    * Class 2: 4 successful sessions; chance of success is Critical Thinking / 12
    * Class 3: 8 successful sessions; chance of success is Critical Thinking / 16

Learning Costs Per Session:
    - Kubani residents subtract 1 coin.
    - Note: "first" refers to the first ancient knowledge that you begin studying in that class bracket.
    - Note 2: The exclusive ancient knowledges have unique costs (they don't fall into the standard class cost).
    - Note 3: Reminder, these are per session costs - not complete cost! Furthermore, you are only charged for successful sessions.
    
    * [Class 1] First: 1 coin, Remaining: 2 coins, Minor Treading: 3 coins.
    * [Class 2] First: 2 coins, Remaining: 3 coins, Horse Riding: 4 coins (then optin to purchase horse for 20 coins after learning).
    * [Class 3] First: 3 coins, Remaining: 4 coins, Major Treading: 5 coins.
    
Learning Benefits:
    * Loot: +15% Better Drops per class (+2VP at class 3)
    * Food: +1 Food Improvement per class (+2VP at class 3)
    * Fatigue: +1 Maximum Fatigue per class (+2VP at class 3)
    * Minor Treading: +2 Minor Actions + 1VP
    * Horse Riding: +3 Road Movements with a Horse + 1VP
    * Major Treading: +1 Major Action + 2VP
"""
        },
    'tamariza':
        {'hallmark': 'Wizard Tower',
         'attribute': 'wizard_tower',
         'description': 
"""You can subscribe to membership with the Wizard Tower for three mystical benefits. Membership period lasts 5 rounds and can be auto-renewed.

    - Tamariza-born citizens get 1 coin discount!
    * Basic Membership: 1 coin
    * Gold Membership: 4 coins 
    * Platinum Membership: 9 coins
    
    1) Teleportation (starting at the Wizard Tower; takes minor action):
        * Basic: Teleport up to 3 tiles away | costs 1 fatigue.
        * Gold: Teleport up to 7 tiles away | costs (tile distance)/3 fatigue, rounded up.
        * Platinum: Teleport anywhere | costs (tile distance)/3.5 fatigue, rounded up.
    2) Matter Conversion (only at the Wizard Tower; takes minor action):
        > G1 (3 categories): [raw meat, raw fish] | [string, beads, hide, sand] | [lead, tin, copper, iron]
        > G2 (2 categories): [cooked meat, cooked fish] | [clay, scales, leather, bark]
        > G3 (3 categories): [well cooked meat, well cooked fish] | [tantalum, aluminum, kevlium, nickel] | [ceramic]
        > G4 (2 categories): [tungsten, diamond, titanium, chromium] | [rubber]
        * Basic: Can convert same category material only in G1.
        * Gold: Can convert same category material in all groups.
        * Platinum: Can convert same and cross-category material in all groups.
    3) Tongue Enchantment (used anywhere):
        > Bartering and Persuasion increase their chance of success.
        * Basic: 1/3 chance of using 1 level higher.
        * Gold: 90% chance for 1 level higher.
        * Platinum: Either 1 or 2 levels higher with equal probability."""
        },
}

# Benfriege
library_level_map = {'Beginner': {'skill': [0, 3], 'min_prob': 0.8, 'min_ct': 0, 'max_ct': 4, 'cost': 2, 'xp': 3}, 
                     'Intermediate': {'skill': [4, 5], 'min_prob': 0.5, 'min_ct': 1, 'max_ct': 8, 'cost': 3, 'xp': 4},
                     'Advanced': {'skill': [6, 7], 'min_prob': 0.2, 'min_ct': 2, 'max_ct': 12, 'cost': 4, 'xp': 4}}

# Kubani
kubani_class_effect = {'loot_class': {0: 'None', 1: '15% Better Drops', 2: '30% Better Drops', 3: '45% Better Drops & 2VP'},
                       'food_class': {0: 'None', 1: '+1 Food Benefit', 2: '+2 Food Benefit', 3: '+3 Food Benefit & 2VP'},
                       'fatigue_class': {0: 'None', 1: '+1 Max Fatigue', 2: '+2 Max Fatigue', 3: '+3 Max Fatigue & 2VP'},
                       'minor_treading': {False: 'None', True: '+2 Minor Actions & 1VP'},
                       'horse_riding': {False: 'None', True: '+3 Road Moves with Horse & 1VP'},
                       'major_treading': {False: 'None', True: '+1 Major Action & 2VP'}}
kubani_req = {'loot_class': 'Gathering', 'food_class': 'Heating', 'fatigue_class': 'Survival'}
kubani_req_lvl = {0: 2, 1: 6, 2: 10}
kubani_exclusive_req = {'minor_treading': 15, 'horse_riding': 45, 'major_treading': 80}
kubani_progress_required = {0: 2, 1: 4, 2: 8, 3: 'Complete'}
kubani_exclusive_progress_required = {'minor_treading': 2, 'horse_riding': 4, 'major_treading': 8}
kubani_cost = {0: 2, 1: 3, 2: 4}
kubani_exclusive_cost = {'minor_treading': 3, 'horse_riding': 4, 'major_treading': 5}
kubani_max_lvl = {'loot_class': 3, 'food_class': 3, 'fatigue_class': 3, 'minor_treading': True, 'major_treading': True, 'horse_riding': True}
kubani_exclusive_class = {'minor_treading': 0, 'horse_riding': 1, 'major_treading': 2}
kubani_ct_den = {0: 8, 1: 12, 2: 18}
horse_cost = 20

# Tamariza
membership_price = {None: 0, 'Basic': 1, 'Gold': 4, 'Platinum': 9}
membership_order = [None, 'Basic', 'Gold', 'Platinum']
matter_conversion_groupers = {1: [{'raw meat', 'raw fish'}, {'string', 'beads', 'hide', 'sand'}, {'lead', 'tin', 'copper', 'iron'}],
                              2: [{'cooked meat', 'cooked fish', 'fruit'}, {'clay', 'scales', 'leather', 'bark'}],
                              3: [{'well cooked meat', 'well cooked fish'}, {'tantalum', 'aluminum', 'kevlium', 'nickel'}, {'ceramic'}],
                              4: [{'tungsten', 'diamond', 'titanium', 'chromium'}, {'rubber'}]}
matter_conversion_items = {}
matter_conversion_cross = {}
for group, categories in matter_conversion_groupers.items():
    matter_conversion_cross[group] = set()
    for i, category in enumerate(categories):
        for item in category:
            matter_conversion_items[item] = {'group': group, 'category': i}
        matter_conversion_cross[group] = matter_conversion_cross[group].union(category)
matter_conversion_rules = {'Basic': {'gr': {1}, 'type': 'same'},
                           'Gold': {'gr': {1, 2, 3, 4}, 'type': 'same'},
                           'Platinum': {'gr': {1, 2, 3, 4}, 'type': 'cross'}}
max_teleport_distance = {'Basic': 3, 'Gold': 7, 'Platinum': float('inf')}

# Zinzibar
zinzibar_class_benefit = {'sharpness': '+1 Attack per round', 
                          'invincibility': 'Survive for another attack if HP goes to 0',
                          'vanish': "Skip opponent's attack round",
                          'shadow': 'Convert a block to dodge',
                          'vision': 'Ignore all fake attack per round'}
zinzibar_class_effect = {'sharpness': {0: 'None', 1: '5%', 2: '10%', 3: '20%'},
                         'invincibility': {0: 'None', 1: '10%', 2: '20%', 3: '40%'},
                         'vanish': {0: 'None', 1: '2%', 2: '4%', 3: '8%'},
                         'shadow': {0: 'None', 1: '3%', 2: '6%', 3: '12%'},
                         'vision': {0: 'None', 1: '4%', 2: '8%', 3: '16%'}}
zinzibar_req = {'sharpness': 'Technique', 'invincibility': 'Hit Points', 'vanish': 'Agility', 'shadow': 'Stability', 'vision': 'Cunning'}
zinzibar_req_lvl = {0: 2, 1: 6, 2: 10}
zinzibar_progress_required = {0: 2, 1: 4, 2: 8, 3: 'Complete'}
zinzibar_max_lvl = 3
zinzibar_ct_den = {0: 8, 1: 12, 2: 18}
