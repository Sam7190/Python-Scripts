# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:12:39 2024

@author: samir
"""

import gameVariables as var
from gameVariables import hallmarks as hvb
from common_widgets import Table
from essentialfuncs import Test_Player

import numpy as np
from time import time
from functools import partial

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
        pass

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
        consequence = self.player.takeDamage(0, self.player.grand_library['read_fatigue'])
        
        if not consequence:
            self.player.exitActionLoop('minor')()
        else:
            # If the player is paralyzed or fainted, then no reason to take the minor action following that.
            self.player.go2action()
    
    def checkout(self, book, _=None):
        if self.player.paused:
            return
        # We assume the warning has already been handled in self.choice
        self.player.addItem(book, 1, skip_update=True)
        self.update_field('total_borrowed', '+1')
        if book in self.player.grand_library['borrowed']:
            self.player.grand_library['borrowed'][book] += 1
        else:
            self.player.grand_library['borrowed'][book] = 1
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
        loan_length = self.get_loan_length()
        self.player.output(f"Receiving a loan of {loan_amount} coins for maximum duration of {loan_length}.", 'green')
        self.update_field('loan_amount', loan_amount)
        self.update_field('loan_length_remaining', loan_length)
        self.update_field('original_loan_length', loan_length)
        self.player.add_coins(loan_amount)
        self.player.exitActionLoop('minor')()
    
    def get_loan(self, _=None):
        if self.player.paused:
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
                self.player.output(f"You did not gain any credit score because you returned your loan early", 'yellow')
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
        pass
    
class fodker:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass

class glaser:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class kubani:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class pafiz:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class scetcher:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class starfex:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class tamarania:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
class tamariza:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass

class tutalu:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass

class zinzibar:
    def __init__(self, player):
        self.player = player
        self.actionFuncs = {}
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
        
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