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
    def __init__(self, lvl