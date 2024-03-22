# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:12:39 2024

@author: samir
"""

import gameVariables as var
from gameVariables import hallmarks as hvb
from common_widgets import Table
from essentialfuncs import Test_Player, hex_distance

import numpy as np
from time import time
from functools import partial
from copy import deepcopy

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
#from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
#from kivy.uix.relativelayout import RelativeLayout
#from kivy.uix.stacklayout import StackLayout
#from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
#from kivy.uix.textinput import TextInput
# to use buttons:
#from kivy.uix.widget import Widget
from kivy.uix.button import Button
#from kivy.uix.image import Image
#from kivy.graphics import Color,Rectangle,Ellipse,InstructionGroup
#from kivy.uix.behaviors import ButtonBehavior
#from kivymd.uix.behaviors import HoverBehavior
#from kivy.uix.screenmanager import ScreenManager, Screen
        
class anafola:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return

def fibonacci(n):
    sqrt_5 = np.sqrt(5)
    phi = (1 + sqrt_5) / 2
    return np.round(((phi ** n) - ((-phi) ** -n)) / sqrt_5)

class benfriege:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
        self.current_round = None
        self.current_books = set()
        
        self.foi = ['read_fatigue', 'read_action_counter', 'total_borrowed', 'books_learned']
        data = [[k.replace('_', ' ').title(), self.player.grand_library[k]] for k in self.foi]
        
        self.table = Table(header=['Field', 'Value'], 
                           data=data,
                           color_odd_rows=True,
                           key_field='Field')
        
    def actionGrid(self):
        self.actionFuncs['Cancel'] = self.player.exitActionLoop(amt=0) # add_back does not work at the moment.
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs)
        
    def update_field(self, field, value):
        if type(value) is str:
            self.player.grand_library[field] += int(value)
        else:
            self.player.grand_library[field] = value
        if field in self.foi:
            k = field.replace('_', ' ').title()
            self.table.update_cell('Value', k, self.player.grand_library[field])
    
    def begin_action_loop(self, _=None):
        # Get Books
        if self.current_round != self.player.round_num:
            self.current_round = self.player.round_num
            self.current_books = set(np.random.choice(var.library_books, 6, False))
        self.actionFuncs = {book: partial(self.choice, book) for book in self.current_books}
        self.actionFuncs['Book Return'] = self.book_return
        self.actionGrid()
    
    def return_book(self, book, _=None):
        if self.player.paused:
            return
        self.player.addItem(book, -1, skip_update=True)
        self.player.grand_library['borrowed'][book] -= 1
        self.update_field('total_borrowed', '-1')
        if self.player.grand_library['borrowed'][book] <= 0:
            self.player.grand_library['borrowed'].pop(book)
        if book == 'learned library book':
            self.player.updateTitleValue('scholar', 1)
        self.player.exitActionLoop('minor')()
    
    def book_return(self, _=None):
        self.actionFuncs = {}
        for item in self.player.items:
            if item in var.lib_book_dict:
                self.actionFuncs[item] = partial(self.return_book, item)
        self.actionGrid()
        
    def extract(self, book):
        first_space = book.index(' ')
        level, skill = book[:first_space].title(), book[(first_space+1):].title()
        warning = False
        if self.player.skills[skill] < var.library_level_map[level]['skill'][0]:
            self.player.output(f'Your {skill} level is too low to learn from this {level} book!', 'yellow')
            warning = True
        elif self.player.skills[skill] > var.library_level_map[level]['skill'][1]:
            self.player.output(f'Your {skill} level is too high to learn from this {level} book!', 'yellow')
            warning = True
        return level, skill, warning
    
    def choice(self, book, _=None):
        level, skill, warning = self.extract(book)
        if not warning:
            self.actionFuncs = {'Read': partial(self.read, book),
                                'Checkout': partial(self.checkout, book)}
            self.actionGrid()
    
    def read(self, book, _=None):
        if self.player.paused:
            return
        if book not in self.current_books:
            self.player.output(f"{book} book is not longer available!", 'yellow')
            return
        level, skill, warning = self.extract(book)
        # The assumption is that the warning has already been handled in self.choice
        ct = self.player.activateSkill("Critical Thinking")
        r = self.player.rbtwn(var.library_level_map[level]['min_ct'], var.library_level_map[level]['max_ct'], None, ct, 'Critical Thinking ')
        if r <= ct:
            self.player.useSkill("Critical Thinking")
            self.player.addXP(skill, 1)
        else:
            self.player.output("You failed to understand the book.", 'red')
        self.update_field('read_fatigue', fibonacci(self.player.grand_library['conseq_read_counter']))
        self.player.grand_library['conseq_read_counter'] += 1
        self.update_field('read_action_counter', 3)
        self.current_books.remove(book)
        consequence = self.player.takeDamage(0, self.player.grand_library['read_fatigue'])
        
        if not consequence:
            self.player.exitActionLoop('minor')()
        else:
            # If the player is paralyzed or fainted, then no reason to take the minor action following that.
            self.player.go2action()
    
    def checkout(self, book, _=None):
        if self.player.paused:
            return
        if book not in self.current_books:
            self.player.output(f"{book} book is not longer available!", 'yellow')
            return
        # We assume the warning has already been handled in self.choice
        self.player.addItem(book, 1, skip_update=True)
        self.update_field('total_borrowed', '+1')
        if book in self.player.grand_library['borrowed']:
            self.player.grand_library['borrowed'][book] += 1
        else:
            self.player.grand_library['borrowed'][book] = 1
        self.current_books.remove(book)
        self.player.exitActionLoop('minor')()

class demetry:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
        self.foi = ['credit_score', 'loan_amount', 'original_loan_length', 'loan_length_remaining', 'strikes', 'total_strikes']
        data = [[k.replace('_', ' ').title(), str(self.get(k))] for k in self.foi]
        
        self.table = Table(header=['Field', 'Value'], 
                           data=data,
                           color_odd_rows=True,
                           key_field='Field')
        
    def actionGrid(self):
        self.actionFuncs['Cancel'] = self.player.exitActionLoop(amt=0) # add back does not work at the moment.
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs)
    
    def get(self, field):
        return self.player.grand_bank[field]
    
    def update_field(self, field, value):
        if type(value) is str:
            self.player.grand_bank[field] += int(value)
        else:
            self.player.grand_bank[field] = value
        if field in self.foi:
            k = field.replace('_', ' ').title()
            self.table.update_cell('Value', k, self.player.grand_bank[field])
        if hasattr(self, field):
            getattr(self, field)(self.player.grand_bank[field])
    
    def end_round(self):
        if self.player.grand_bank['loan_length_remaining'] is not None:
            self.update_field('loan_length_remaining', '-1')
    
    def failure(self):
        self.player.output("Strike 3!! The bank is now automatically deducting your coins and refusing you future service.", 'red')
        self.update_field('credit_score', -1)
        if self.get('loan_amount') <= self.player.coins:
            self.player.add_coins(-self.get('loan_amount'))
            self.player.output(f"{self.get('loan_amount')} coins has been deducted.", 'yellow')
        else:
            still_owe = self.get('loan_amount') - self.player.coins
            self.player.output(f"{self.player.coins} coins has been deducted, but you still owe {still_owe} coins!", 'red')
            self.player.add_coins(-self.player.coins)
            for city in var.home_price_order:
                if self.player.homes[city]:
                    homecost = var.capital_info[city]['home']
                    # begin losing home.
                    self.player.homes[city] = False
                    self.player.max_capacity -= var.capital_info[city]['capacity']
                    self.player.Capital -= var.capital_info[city]['home_cap']
                    self.updateTotalVP(-var.capital_info[city]['home_cap'], False)
                    if not self.player.markets[city]:
                        self.player.market_allowed[city] = False
                    self.player.output(f"You have lost your home in {city}!")
                    if self.player.Reputation < (3 * var.city_info[city]['entry']):
                        self.player.training_allowed[city] = False
                        self.player.output(f"You can no longer use trainers in {city}.", 'red')
                    if homecost > still_owe:
                        var.capital_info[city]['home'] = (homecost - still_owe)
                        self.player.output(f"Next time you purchase the home in {city} it will cost you {homecost-still_owe} coins.")
                        return
                    elif homecost == still_owe:
                        return
                    still_owe -= homecost
            self.player.output(f"The {still_owe} coins you still owe the bank is forgiven.", 'blue')
    
    def loan_length_remaining(self, llr):
        if llr == None:
            return
        elif llr == 0:
            strikes = self.player.grand_bank['strikes']
            if (self.player.grand_bank['credit_score'] <= 0) or (strikes >= 2):
                self.failure()
                return
            self.update_field('strikes', '+1')
            self.update_field('total_strikes', '+1')
            pr = self.player.activateSkill('Persuasion')
            r = self.player.rbtwn(0, 12 if strikes==1 else 18, None, pr, 'Persuasion')
            strikemsg = f"Strike {strikes+1}! You failed to pay back your loan ontime"
            if r <= pr:
                self.player.output(strikemsg + ", but your Persuasion prevented you from losing credit score.", 'red')
                self.player.useSkill('Persuasion')
            else:
                cutpct = 0.25 if strikes==1 else 0.75
                cut = np.min([self.player.grand_bank['credit_score'], int(np.ceil(np.round(cutpct * self.player.grand_bank['credit_score'])))])
                self.update_field('credit_score', f'-{cut}')
                self.player.output(strikemsg + f" and you lost {cutpct*100}% of yoru credit score!", 'red')
            self.update_field('loan_length_remaining', 5)
            self.player.output("The bank gives you five more rounds to return the loan. Come to Demetry as soon as possible!", 'yellow')
        elif llr <= 3:
            self.player.output(f"Only {llr} rounds left on your loan!", 'yellow')
    
    def credit_score(self, cs):
        self.player.updateTitleValue('superprime', None, cs)
            
    def get_loan_options(self, max_options=7):
        if self.player.grand_bank['credit_score'] == -1:
            self.player.output("You are not allowed to take loans anymore!", 'yellow')
            return
        max_loan = self.player.grand_bank['credit_score'] + (self.player.Reputation // 2) + 1
        if max_loan <= 0:
            self.player.output("You should try increasing your reputation to take out a loan!", 'yellow')
            return
        min_loan = np.max([1, max_loan // max_options])
        loan_options = sorted(set(np.linspace(min_loan, max_loan, max_options).round().astype(int)))
        return loan_options
    
    def get_loan_length(self):
        return 10 + self.player.skills['Persuasion']
    
    def get_loan_confirmed(self, loan_amount, _=None):
        if self.player.paused:
            return
        loan_length = self.get_loan_length()
        self.player.output(f"Receiving a loan of {loan_amount} coins for maximum duration of {loan_length}.", 'green')
        self.update_field('loan_amount', loan_amount)
        self.update_field('loan_length_remaining', loan_length)
        self.update_field('original_loan_length', loan_length)
        self.player.add_coins(loan_amount)
        self.player.exitActionLoop('minor')()
    
    def get_loan(self, _=None):
        if self.player.puased:
            return
        loan_options = self.get_loan_options()
        loan_length = self.get_loan_length()
        self.player.output(f'Here are your loan options for a period of {loan_length} rounds.', 'blue')
        self.actionFuncs = {str(lo): partial(self.get_loan_confirmed, lo) for lo in loan_options}
        self.actionGrid()
        
    def return_loan(self, _=None):
        if self.player.paused:
            return
        # We assume that the check whether the player has a loan has been done prior to hitting this button
        if self.player.coins < self.get('loan_amount'):
            self.player.output(f"You must repay the entire balance - and you currently do not have {self.get('loan_amount')} coins to do so.", 'yellow')
            return
        self.player.output(f"You are returning {self.get('loan_amount')} to the bank.")
        if self.get('strikes') <= 0:
            rounds_passed = self.get('original_loan_length') - self.get('loan_length_remaining')
            if rounds_passed >= 5:
                self.update_field('credit_score', f"+{self.get('loan_amount')}")
                self.player.output(f"You gained {self.get('loan_amount')} credit score!", 'green')
            else:
                self.player.output("You did not gain any credit score because you returned your loan early", 'yellow')
        # Complete the return
        self.player.add_coins(-self.get('loan_amount'))
        self.update_field('loan_amount', None)
        self.update_field('loan_length_remaining', None)
        self.update_field('original_loan_length', None)
        
        self.player.exitActionLoop('minor')()
    
    def begin_action_loop(self, _=None):
        self.actionFuncs = {}
        if self.get('loan_amount') is not None:
            self.actionFuncs['Return'] = self.return_loan
        else:
            self.actionFuncs['Take Loan'] = self.get_loan
        self.actionGrid()
    
class enfeir:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class fodker:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return

class glaser:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class kubani:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
        self.foi = ['loot_class', 'food_class', 'fatigue_class', 'minor_treading', 'horse_riding', 'major_treading']
        data = [[k.replace('_', ' ').title(), str(self.get(k)), var.kubani_class_effect[k][self.get(k)]] for k in self.foi]
        
        self.table = Table(header=['Field', 'Value', 'Benefit', 'Successful Sessions', 'Sessions Needed'], 
                           data=data,
                           color_odd_rows=True,
                           key_field='Field')
        
    def actionGrid(self, add_cancel=True):
        if add_cancel:
            self.actionFuncs['Cancel'] = self.player.exitActionLoop(amt=0)
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs)
    
    def get(self, field):
        return self.player.ancestrial_order[field]
    
    def update_field(self, field, value):
        previous_value = deepcopy(self.get(field))
        if type(value) is str:
            try:
                self.player.ancestrial_order[field] += int(value)
            except ValueError:
                # This means that we are not trying to add to the field, rather trying to set it.
                self.player.ancestrial_order[field] = value
        else:
            self.player.ancestrial_order[field] = value
        next_value = self.get(field)
        split = field.split('_')
        if field in self.foi:
            k = field.replace('_', ' ').title()
            self.table.update_cell('Value', k, next_value)
            self.table.update_cell('Benefit', var.kubani_class_effect[field][next_value])
        elif split[-1] == 'progress':
            foi = '_'.join(split[:-1])
            self.table_update_cell('Successful Session', foi, next_value)
            self.check_level_up(foi, next_value)
        if hasattr(self, field):
            getattr(self, field)(previous_value, next_value)
            
    def minor_treading(self, bmt, mt):
        if mt:
            self.player.max_minor_actions += 2
            self.player.updateFellowshipVP(1)
        elif bmt:
            self.player.max_minor_action -= 2
            self.player.updateFellowshipVP(-1)
    
    def major_treading(self, bmt, mt):
        if mt:
            self.player.max_major_actions += 1
            self.player.updateFellowshipVP(2)
        elif bmt:
            self.player.max_major_actions -= 1
            self.player.updateFellowshipVP(-2)
    
    def fatigue_class(self, bfc, fc):
        if fc > bfc:
            self.player.max_fatigue += 1
        elif bfc > fc:
            self.player.max_fatigue -= 1
    
    def loot_class(self, blc, lc):
        if lc > blc:
            self.player.loot_bonus += 0.15
        elif blc > lc:
            self.player.loot_bonus -= 0.15
    
    def food_class(self, bfc, fc):
        if fc > bfc:
            self.player.food_bonus += 1
        elif bfc > fc:
            self.player.food_bonus -= 1
    
    def check_level_up(self, field, progress):
        if progress == 'Complete':
            return
        if field in var.kubani_exclusive_progress_required:
            if progress >= var.kubani_exclusive_progress_required[field]:
                self.player.output(f"You successfully completed {' '.join(field.split(' ')).title()} learning!", 'green')
                self.update_field(field, True)
                self.player.output(f"You now have {var.kubani_class_effect[field][True]}.", 'blue')
                self.update_field(field+'_progress', 'Complete')
                self.table_update_cell('Sessions Needed', field, 'Complete')
        else:
            lvl = self.get(field)
            if progress >= var.kubani_progress_required[lvl]:
                self.player.output(f"You progressed to {' '.join(field.split('_')).title()} {lvl+1}!", 'green')
                self.update_field(field, '+1')
                if (lvl+1) == var.kubani_max_lvl[field]:
                    self.player.updateFellowshipVP(2)
                self.player.output(f"You now have {var.kubani_class_effect[field][lvl+1]}.", 'blue')
                value = var.kubani_progress_required[self.get(field)]
                self.table_update_cell('Sessions Needed', field, value)
                restart = 'Complete' if value == 'Complete' else 0
                self.update_field(field+'_progress', restart)
                
    def confirm_horse_purchase(self, cost, _=None):
        if self.player.paused:
            return
        if self.player.coins < cost:
            self.insufficient_funds(cost)
            return
        self.player.add_coins(-cost)
        self.player.has_horse = True
        self.player.max_road_moves += 3
        self.player.output("You now own a horse! You get +3 road movements.", 'green')
        self.player.exitActionLoop('purchase', 1)()
                
    def purchase_horse(self, _=None):
        if self.player.paused:
            return
        cost = var.horse_cost - self.player.bartering_mode
        if self.player.coins < cost:
            self.insufficient_funds(cost)
            return
        self.player.output(f"Purchasing a horse will cost you {cost}. Make the purchase?", 'blue')
        self.actionFuncs = {'*g|Confirm Purchase': partial(self.confirm_horse_purchase, cost), 'Cancel': self.player.output('purchase', 0)}
        self.actionGrid(add_cancel=False)
        
    def init_purchase_horse(self, _=None):
        if self.player.paused:
            return
        if self.player.has_horse:
            self.player.output("You already own a horse!", 'yellow')
            return
        if not self.player.activated_bartering:
            self.player.output(f"Horses cost {var.horse_cost}. Attempt to Barter?", 'blue')
            self.actionFuncs = {'Attempt to Barter': partial(self.player.attempt_barter, self.purchase_horse), "Don't Barter": self.purchase_horse}
            self.actionGrid()
        else:
            self.purchase_horse()
    
    def get_study_cost(self, field):
        if field in var.kubani_exclusive_cost:
            max_cost = var.kubani_exclusive_cost[field]
        else:
            lvl = self.get(field)
            first_class = self.get(f'first_class_{lvl+1}')
            max_cost = var.kubani_cost[lvl]
            if (first_class is None) or (first_class == field):
                max_cost -= 1
        if self.player.birthcity=='kubani':
            max_cost -= 1
        if max_cost >= 2:
            max_cost -= min([1, self.player.bartering_mode])
        else:
            max_cost = max([0, max_cost])
        return max_cost
    
    def get_study_req_met(self, field, display=True):
        if field in var.kubani_exclusive_req:
            lvl = self.player.Knowledge
            req = var.kubani_exclusive_req[field]
            if self.player.birthcity=='kubani':
                req -= 5
            t = 'Total Knowledge'
        else:
            t = var.kubani_req[field]
            lvl = self.player.skills[t]
            req = var.kubani_req_lvl[self.get(field)]
            if self.player.birthcity=='kubani':
                req -= 1
        if lvl < req:
            if display:
                self.player.output(f"Your {t} Level is too low! You need at least {req}.", 'yellow')
            met = False
        else:
            met = True
        return met
    
    def insufficient_funds(self, cost):
        self.player.output(f"You do not have {cost} coins!", 'yellow')
        self.player.exitActionLoop('purchase', 0)()
    
    def attempt_study(self, field, cost, _=None):
        if self.player.paused:
            return
        if self.player.coins < cost:
            self.insufficient_funds(cost)
            return
        c = var.kubani_exclusive_class[field] if field in var.kubani_exclusive_class else self.get(field)
        ct_den = var.kubani_ct_den[c]
        ct = self.player.activateSkill('Critical Thinking')
        r = self.player.rbtwn(1, ct_den, None, ct, 'Critical Thinking ')
        if r <= ct:
            self.player.useSkill("Critical Thinking")
            self.player.output("You successfully understood the session!", 'green')
            self.update_field(field+'_progress', '+1')
            self.player.add_coins(-cost)
        else:
            self.player.output("You failed to understand the session.", 'yellow')
        self.player.exitActionLoop(amt=1)()
    
    def study(self, field, _=None):
        if self.player.paused:
            return
        cost = self.get_study_cost(field)
        if self.player.coins < cost:
            self.insufficient_funds(cost)
            return
        self.player.output(f"One successful study session for {self.get_study_name(field)} will cost {cost}. Attempt to Study?", 'blue')
        self.actionFuncs = {'*g|Attempt Study': partial(self.attempt_study, field, cost), 'Cancel': self.player.exitActionLoop('purchase', 0)}
        self.actionGrid(add_cancel=False)
    
    def init_study(self, field, _=None):
        if self.player.paused:
            return
        if not self.get_study_req_met(field):
            return
        if self.player.skills['Critical Thinking'] < 1:
            self.player.output("You need at least Level 1 Critical Thinking to learn!", 'yellow')
            return
        if not self.player.activated_bartering:
            cost = self.get_study_cost(field)
            self.player.output(f"Studying {self.get_study_name(field)} will cost {cost}. Attempt to Barter?", 'blue')
            self.actionFuncs = {'Attempt to Barter': partial(self.player.attempt_barter, partial(self.study, field)),
                                "Don't Barter": partial(self.study, field)}
            self.actionGrid(add_cancel=False)
        else:
            self.study(field)
    
    def get_study_name(self, field):
        split = field.split('_')
        if split[-1] == 'class':
            split.pop()
        study = f"Study {' '.join(split).title()}"
        return study
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
        self.actionFuncs = {}
        for field in self.foi:
            if self.get(field) >= var.kubani_max_lvl[field]:
                continue
            if not self.get_study_req_met(field, False):
                continue
            study = self.get_study_name(field)
            self.actionFuncs[study] = partial(self.init_study, field)
        if self.get('horse_riding') and (not self.player.has_horse):
            self.actionFuncs['Purchase Horse'] = self.init_purchase_horse
        self.actionGrid()
    
class pafiz:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class scetcher:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class starfex:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class tamarania:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
    
class tamariza:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
        self.foi = ['membership', 'membership_rounds_remaining', 'auto_renewal', 'renew_with_market_earnings', 'current_consecutive_platinum_renewals', 'best_consecutive_platinum_renewals']
        data = [[k.replace('_', ' ').title(), str(self.get(k))] for k in self.foi]
        
        self.table = Table(header=['Field', 'Value'], 
                           data=data,
                           color_odd_rows=True,
                           key_field='Field')
        
    def actionGrid(self, add_cancel=True):
        if add_cancel:
            self.actionFuncs['Cancel'] = self.player.exitActionLoop(amt=0) # add back does not work at the moment.
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs)
    
    def get(self, field):
        return self.player.wizard_tower[field]
    
    def update_field(self, field, value):
        previous_value = self.player.wizard_tower[field]
        if type(value) is str:
            try:
                self.player.wizard_tower[field] += int(value)
            except ValueError:
                # This means that we are not trying to add to the field, rather trying to set it.
                self.player.wizard_tower[field] = value
        else:
            self.player.wizard_tower[field] = value
        if field in self.foi:
            k = field.replace('_', ' ').title()
            self.table.update_cell('Value', k, self.player.wizard_tower[field])
        if hasattr(self, field):
            getattr(self, field)(previous_value, self.get(field))
    
    def end_round(self):
        if self.get('membership_rounds_remaining') is not None:
            self.update_field('membership_rounds_remaining', '-1')
    
    def membership(self, pm, m):
        if m is None:
            self.update_field('auto_renewal', 'off')
            self.update_field('membership_rounds_remaining', None)
        if m != 'Platinum':
            self.update_field('current_consecutive_platinum_renewals', 0)
        if (pm == 'Platinum') and (m == 'Platinum'):
            self.update_field('current_consecutive_platinum_renewals', '+1')
        color = 'green' if var.membership_order.index(m) >= var.membership_order.index(pm) else ('yellow' if pm is None else None)
        msg = f"You successfully subsribed to {m} Wizard Tower membership!" if pm is None else "Your Wizard Tower membership has changed to {m}."
        self.player.output(msg, color)
        if (m is not None) and (self.get('auto_renewal') == 'off'):
            self.actionFuncs = {'Turn Auto Renewal on': self.toggle_auto_renewal, 'Keep Auto Renewal off': self.player.exitActionLoop(amt=0)}
            self.actionGrid(add_cancel=False)
    
    def current_consecutive_platinum_renewals(self, pccpr, ccpr):
        if ccpr > self.get('best_consecutive_platinum_renewals'):
            self.update_field('best_consecutive_platinum_renewals', ccpr)
        
    def best_consecutive_platinum_renewals(self, pbcpr, bcpr):
        self.player.updateTitleValue('sorcerer', None, bcpr)
            
    def membership_rounds_remaining(self, pmrr, mrr):
        if mrr == 0:
            if self.get('auto_renewal') == 'off':
                qm = self.get('queued_membership')
                self.update_field('membership', qm)
            elif self.get('auto_renewal') == 'on':
                qm = self.get('queued_membership')
                if qm is None:
                    # Go to auto-renewal with using market coins first if turned on.
                    cost = var.membership_price[self.get('membership')]
                    if self.player.birthcity.lower() == 'tamariza':
                        cost -= 1
                    if self.get('renew_with_market_earnings') == 'off':
                        if self.player.coins < cost:
                            self.player.output(f"You do not have the {cost} coins with you to renew your membership!", 'red')
                            self.update_field('membership', None)
                        else:
                            self.player.add_coins(-cost)
                            self.update_field('membership_rounds_remaining', 5)
                            self.player.output(f"The Wizard Tower auto-deducted {cost} coins from you to renew your membership.", 'green')
                    else:
                        if (self.player.bank['tamariza'] + self.player.coins) < cost:
                            self.player.output("You do not have enough coins with you and at your Demetry market to renew your membership!", 'red')
                            self.update_field('membership', None)
                        else:
                            withdraw = min([self.player.bank['tamariza'], cost])
                            self.player.bank['tamariza'] -= withdraw
                            self.player.output(f"Withdrew {withdraw} from you market earnings to pay for your Wizard Tower auto-renewal membership", 'green')
                            if withdraw < cost:
                                self.player.add_coins(-(cost-withdraw))
                                self.player.output(f"The Wizard Tower also auto-deducted {cost-withdraw} coins to complete your {self.get('membership')} membership renewal", 'green')
                            self.update_field('membership_rounds_remaining', 5)
                else:
                    # This means they must have purchased a greater membership already, so turn it on.
                    self.update_field('membership', qm)
                    self.update_field('membership_rounds_remainig', 5)
    
    def confirm_membership(self, membership, cost, _=None):
        if self.player.paused:
            return
        if membership == None:
            self.update_field('queued_membership', None)
            self.update_field('auto_renewal', 'off')
            self.player.player.output(f"Your membership will cancel in {self.get('membership_rounds_remaining')} rounds", 'green')
        else:
            self.player.add_coins(-cost)
            if self.get('membership') is None:
                self.update_field('membership', membership)
                self.update_field('membership_rounds_remaining', 5)
            else:
                self.update_field('queued_membership', membership)
        self.player.exitActionLoop('minor')()
    
    def cancel(self, _=None):
        if self.player.paused:
            return
        self.actionFuncs = {}
        self.player.output("When you cancel you keep your membership until the end and all renewals will be turned off. Proceed?", 'blue')
        self.actionFuncs['*g|Confirm'] = partial(self.confirm_membership, None, 0)
        self.actionGrid()
        
    def begin_membership(self, membership, _=None):
        if self.player.paused:
            return
        cost = var.membership_price[membership]
        if self.player.birthcity.lower() == 'tamariza':
            cost -= 1
        if self.player.coins < cost:
            self.player.output(f"You do not have {cost} coins to purchase this membership!", 'yellow')
            return
        self.player.output(f'Confirm {membership} membership for {cost} coin{"s" if cost!=1 else ""}', 'blue')
        if self.get('membership') is not None:
            self.player.output("Note! Your new membership will not start until you complete your old one.", 'yellow')
        self.actionFuncs = {'*g|Confirm': partial(self.confirm_membership, membership, cost)}
        self.actionGrid()
    
    def change_membership(self, _=None):
        if self.player.paused:
            return
        self.actionFuncs = {}
        cm = var.membership_order.index(self.get('membership')) + 1
        for m in var.membership_order[cm:]:
            self.actionFuncs[m] = partial(self.begin_membership, m)
        if self.get('membership') != None:
            self.actionFuncs['Cancel Membership'] = self.cancel
        self.actionGrid()
    
    def cancel_teleport(self, _=None):
        if self.player.paused:
            return
        self.player.teleport_ready = False
        self.player.parentBoard.game_page.main_screen.current = 'City'
        self.player.output("Canceled teleport.")
        self.player.exitActionLoop(amt=0)()
        
    def confirm_teleport(self, coord, fatigue, _=None):
        if self.player.paused:
            return
        self.player.teleport_ready = False
        self.player.takeDamage(0, fatigue)
        self.player.moveto(coord, trigger_consequence=True)
        self.player.exitActionLoop('minor')()
    
    def teleport_to(self, coord):
        if self.player.paused:
            return
        if coord == self.player.currentcoord:
            self.cancel_teleport()
            
        # Do fatigue calculation here
        m = self.get('membership')
        distance = hex_distance(self.player.currentcoord, coord)
        if distance > var.max_teleport_distance[m]:
            self.player.output(f"This is too far for you to teleport with your {m} membership!", 'yellow')
            return
        fatigue = int(np.ceil(distance / 3.5)) if m=='Platinum' else int(np.ceil(distance / 3))
        if (self.player.fatigue + fatigue) >= self.player.max_fatigue:
            self.player.output(f"You can't teleport here, it would cost you {fatigue} which is beyond your capacity at the moment", "yellow")
            return
        
        self.player.output(f"Teleporting to {self.player.parentBoard.gridtiles[coord].tile.title()} will cost {fatigue} fatigue. Do wish to proceed?", 'blue')
        self.actionFuncs = {}
        self.actionFuncs['*g|Yes'] = partial(self.confirm_teleport, coord, fatigue)
        self.actionFuncs['Choose Different Tile'] = self.teleport
        self.actionFuncs['Cancel Teleport'] = self.cancel_teleport
        self.actionGrid(add_cancel=False)
    
    def teleport(self, _=None):
        if self.player.paused:
            return
        self.player.teleport_ready = True
        self.player.parentBoard.game_page.main_screen.current = 'Board'
        self.player.output("Click on a tile to teleport to!", 'blue')
        self.actionFuncs = {'Cancel Teleport': self.cancel_teleport}
        self.actionGrid(add_cancel=False)
    
    def confirm_convert(self, from_item, to_item, _=None):
        if self.player.paused:
            return
        self.player.addItem(from_item, -1, skip_update=True)
        self.player.addItem(to_item, 1, skip_update=True)
        self.player.exitActionLoop('minor')()
    
    def convert_to(self, from_item, to_item, _=None):
        if self.player.paused:
            return
        self.player.output(f"Convert {from_item} to {to_item}?", 'blue')
        self.actionGrid = {"*g|Yes": partial(self.confirm_convert, from_item, to_item)}
        self.actionGrid()
        
    def convert_item(self, item, _=None):
        if self.player.puased:
            return
        g = var.matter_conversion_items[item]['group']
        c = var.matter_conversion_items[item]['category']
        t = var.matter_conversion_rules[self.get('membership')]['type']
        item_set = var.matter_conversion_groupers[g][c] if t=='same' else var.matter_conversion_cross[g]
        convert_to_options = item_set.difference({item})
        self.actionFuncs = {}
        for to_item in convert_to_options:
            self.actionFuncs[to_item] = partial(self.convert_to, item, to_item)
        self.actionGrid()
    
    def matter_conversion(self, _=None):
        if self.player.paused:
            return
        m = self.get('membership')
        gr = var.matter_conversion_rules[m]['gr']
        self.actionFuncs = {}
        for item in self.player.items:
            if var.matter_conversion_items[item]['group'] in gr:
                self.actionFuncs[item] = partial(self.convert_item, item)
        self.actionGrid()
    
    def auto_renewal(self, par, ar):
        if ar == 'off':
            self.update_field('renew_with_market_earnings', 'off')
    
    def toggle_renew_with_market(self, _=None):
        if self.player.paused:
            return
        toggle = 'on' if self.get('renew_with_market_earnings')=='off' else 'on'
        clr = 'green' if toggle=='on' else None
        self.update_field('renew_with_market_earnings', toggle)
        self.player.output(f"Turned renew with market earnings {toggle}.", clr)
        self.player.exitActionLoop(amt=0)()
    
    def toggle_auto_renewal(self, _=None):
        if self.player.paused:
            return
        toggle = 'on' if self.get('auto_renewal') == 'off' else 'off'
        clr = 'green' if toggle=='on' else None
        self.update_field('auto_renewal', toggle)
        self.player.output(f"Turned auto renewal {toggle}.", clr)
        if toggle == 'on':
            self.player.output("Would you also like to turn Renew with Market Earnings on?", 'blue')
            self.actionFuncs = {'Turn Renew with Market Earnings on': self.toggle_renew_with_market, 'Keep Renew with Market Earnings off': self.player.exitActionLoop(amt=0)}
            self.actionGrid(add_cancel=False)
        else:
            self.player.exitActionLoop(amt=0)()
    
    def begin_action_loop(self, _=None):
        self.actionFuncs = {}
        self.actionFuncs['Change Membership'] = self.change_membership
        if self.get('membership') is not None:
            self.actionFuncs['Teleport'] = self.teleport
            self.actionFuncs['Matter Conversion'] = self.matter_conversion
            toggle = 'on' if self.get('auto_renewal') == 'off' else 'off'
            self.actionFuncs[f'Turn Auto Renewal {toggle}'] = self.toggle_auto_renewal
            mtoggle = 'on' if self.get('renew_with_market_earnings')=='off' else 'on'
            if toggle == 'off':
                self.actionFuncs[f'Turn Renew with Market Earnings {mtoggle}'] = self.toggle_renew_with_market
        self.actionGrid()

class tutalu:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return

class zinzibar:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        if self.player.paused:
            return
        
class HallmarkPanels(TabbedPanel):
    def __init__(self, city, player, **kwargs):
        super().__init__(**kwargs)
        self.do_default_tab = True
        self.tabs = {}
        self.widgets = {}
        
        self.description = Button(text=hvb[city]['description'],
                                  disabled=True,
                                  background_normal='',
                                  font_size=12,
                                  halign='left',
                                  valign='top')
        self.add_tab('Description', self.description)
    
    def add_tab(self, name, widget):
        tab = TabbedPanelItem(text = name)
        tab.add_widget(widget)
        self.tabs[name] = tab
        self.widgets[name] = widget
        
    def publish(self, order=None, default=None):
        if order is None:
            for name, tab in self.tabs.items():
                self.add_widget(tab)
        else:
            for name in order:
                self.add_widget(self.tabs[name])
        if default is not None:
            self.default_tab = self.tabs[default]

class Hallmark(FloatLayout):
    def __init__(self, city, player, **kwargs):
        super().__init__(**kwargs)
        self.city = city
        self.hallmark = hvb[city]['hallmark']
        self.player = player
        self.hallmark_widgets = eval(city)(player)
        
        self.title = Label(text=self.hallmark,
                           font_size=36,
                           bold=True,
                           font_name='fonts\\Cinzel-Bold.ttf',
                           pos_hint={'x': 0.02, 'top': 0.925},
                           size_hint=(0.35, 0.08),
                           halign='left',
                           valign='bottom')
        self.add_widget(self.title)
        
        # self.description = Button(text=hvb[city]['description'], 
        #                           disabled=True,
        #                           background_normal='',
        #                          font_size=12,
        #                          pos_hint={'x': 0.02, 'top': 0.85}, 
        #                          size_hint=(0.96, 0.35), 
        #                          halign='left',
        #                          valign='top')
        #self.add_widget(self.description)
        
        self.panels = HallmarkPanels(city, player,
                                     pos_hint={'x': 0.02, 'top': 0.85},
                                     size_hint=(0.96, 0.5))
        
        self.panels.publish()
        self.add_widget(self.panels)
    
class MyApp(App):
    def build(self):
        return Hallmark('benfriege', Test_Player())

if __name__ == '__main__':
    MyApp().run()