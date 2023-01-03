# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:59:23 2021

@author: samir
"""

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
from kivy.clock import Clock
from kivy.core.window import Window


class BarterPage(FloatLayout):
    def __init__(self, lvl, merchant, tile, **kwargs):
        super().__init__(**kwargs)
        self.playerLvl = lvl
        self.progress = 0
        # Lvl 0-3: 2 chances, Lvl 4-7: 3 chances, Lvl 8-11: 4 chances, Lvl 12: 5 chances
        self.chancesLeft = 2 + (lvl // 4)
        self.options = {'Prompt Price':     {'init':    'So how much are you really selling it for?', 
                                                 'regular': 'Prices are as marked.', 
                                                 'failure': "You won't be able to afford our prices.", 
                                                 'success': 'Alright, how are these prices for you?',
                                                 'inspect': "Unlikely to hurt the haggle, but usually doesn't help much either"},
                        
                        'Ask for Deal':       {'init':    'Come on, give me a good deal on these.', 
                                                 'regular': "You can't beat these prices.", 
                                                 'failure': 'How does NO deal sound?', 
                                                 'success': 'You know what, just for you, how does this deal sound?',
                                                 'inspect': "Unlikely to hurt the haggle, but usually doesn't help much either"},
                        
                        'Set Ceiling':          {'init':    'I am not spending more than they are worth.', 
                                                 'regular': 'These are the best prices around', 
                                                 'failure': 'Are you implying that I am trying to cheat you?', 
                                                 'success': 'Okay, how about this?',
                                                 'inspect': 'May hurt the haggle, but could also help.'}, 
                        
                        'Proclaim Value':  {'init':    'You and I both know how much these are trully worth.', 
                                                 'regular': 'Do these prices bother you?', 
                                                 'failure': "If you know how much they are worth, then why don't you sell them?",
                                                 'success': "Do these prices look better?",
                                                 'inspect': 'May hurt the haggle, but could also help.'}, 
                        
                        'Mention a Competitor': {'init':    'Your not the only people I can get these items from, you know?', 
                                                 'regular': 'I know.', 
                                                 'failure': 'So go to them then.', 
                                                 'success': 'Let me bring the prices down a notch.',
                                                 'inspect': 'Aggressive tactic that effects only those under pressure by the competition. Otherwise, could hurt the haggle a lot.'},
                        
                        'Mention a Referral':   {'init':    'A friend told me you are the best. I could continue spreading the word.', 
                                                 'regular': 'Great.', 
                                                 'failure': 'Are you trying to bribe me?', 
                                                 'success': 'Fantastic, hey, how about this discount on me?',
                                                 'inspect': "Doesn't usually hurt or help the haggle, but when it works, it works well."},
                        
                        'Remark on Integrity':  {'init':    'Are you trying to cheat me with these prices?',
                                                 'regular': 'No. This is the market value.',
                                                 'failure': "Are you questioning my market knowledge? You should take your business elswhere if you know so much better.",
                                                 'success': 'Forgive me. Perhaps these prices sound more reasonable?',
                                                 'inspect': 'Powerful threat that usually results in failure. However, when successful, it can be very impactful.'},
                        
                        'Societal Obligation':  {'init':    'Any savings I get here I can use to benefit others in the community.',
                                                 'regular': 'I offer good prices.',
                                                 'failure': "And what of my family?",
                                                 'success': 'I am happy to do whatever will help - how about these prices?',
                                                 'inspect': "Doesn't usually hurt or help the haggle, but when it works, it works well."},
                        
                        'Remain Silent':        {'init':    '...',
                                                 'regular': 'Is there any way I can be of service?',
                                                 'failure': 'If you are just here to loaf around, then go somewhere else!',
                                                 'success': 'How about a reduction in price for you?',
                                                 'inspect': 'This can only fail if the conversation has dragged on too long.'},
                        
                        'Buy in Bulk (3+)':     {'init':    'I wish to buy in bulk - what bulk deals will you give me?',
                                                 'regular': 'My prices are already adjusted to bulk orders.',
                                                 'failure': 'What do I look like to you, a warehouse?',
                                                 'success': 'Certainly, these are my bulk prices.',
                                                 'inspect': 'Usually only helps the haggle - but will require the willingness to buy in bulk.'},
                        
                        'Walk Away':            {'init':    'So be it. [You walk away]', 
                                                 'regular': 'Come on, lets keep negotiating.',
                                                 'failure': 'So long then.', 
                                                 'success': 'Okay, okay, about how these prices?',
                                                 'inspect': 'Classic closing tactic: You will either succeed or fail - any inbetween is rare.'}}