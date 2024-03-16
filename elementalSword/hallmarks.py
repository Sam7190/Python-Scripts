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
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
        
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
        
    def actionGrid(self):
        self.player.parentBoard.game_page.make_actionGrid(self.actionFuncs, add_back=True)
    
    def update_field(self, field, value):
        pass
    
    def begin_action_loop(self, _=None):
        pass
    
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