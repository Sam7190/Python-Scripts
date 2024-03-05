# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 06:45:02 2024

@author: samir
"""

import numpy as np
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.properties import ListProperty
from kivy.core.window import Window

import essentialfuncs as essf

class LockIcon(Image):
    def __init__(self, tile, reputation_required, **kwargs):
        super().__init__(**kwargs)
        self.source = f'images\\icons\\locked\\{reputation_required}.png'
        self.tile = tile
        self.pos_hint = tile.pos_hint
        self.size_hint = tile.size_hint
        self.locked = True
    def delete(self):
        self.source = ''
        self.opacity = 0
        self.locked = False

class SkirmishIcon(Image):
    def __init__(self, tile, stage, **kwargs):
        super().__init__(**kwargs)
        self.set_stage(stage)
        self.tile = tile
        self.pos_hint = tile.pos_hint
        self.size_hint = tile.size_hint
    def set_stage(self, stage):
        self.stage = stage
        if (stage <= 0) or (self.tile.lockIcon.locked):
            self.source = ''
            self.opacity = 0
        else:
            self.source = f'images\\icons\\skirmish\\{stage}.png'
            self.opacity = 0.65

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
        self.header_color = header_color
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
        self.cells, self.header_widgets = {}, {}
        for h in header:
            self.cells[h] = []
            clr = ['[b]','[/b]'] if header_color is None else [f'[color={essf.get_hexcolor(header_color)}][b]','[/b][/color]']
            hkwargs = {} if text_color is None else {'color':text_color}
            if header_as_buttons:
                L = Button(text=clr[0]+h+clr[1],markup=True,background_color=header_bkg_color,underline=True,**hkwargs)
            else:
                L = Label(text=clr[0]+h+clr[1],markup=True,valign='bottom',halign='center',**hkwargs)
                #L.text_size = L.size
                if wrap_text: L.text_size = L.size
            self.header_widgets[h] = L
            self.add_widget(L)
        # In the case that data is one-dimensional, then make it a matrix of one row.
        self.input_text_color = text_color
        self.wrap_text = wrap_text
        self.update_data_cells(data)
    def clear_data_cells(self):
        for h in self.cells:
            for L in self.cells[h]:
                L.text = ''
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
                        self.cells[cell][i].text=item['text']
                        if 'color' in item:
                            self.cells[cell][i].color = item['color']
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
    def update_header(self, header_name, new_value):
        clr = ['[b]','[/b]'] if self.header_color is None else [f'[color={essf.get_hexcolor(self.header_color)}][b]','[/b][/color]']
        self.header_widgets[header_name].text = clr[0] + new_value + clr[1]
    def update_bkgSize(self, instance, value):
        self.bkg.size = self.size
        self.bkg.pos = self.pos
        if self.wrap_text:
            for L in self.children:
                L.text_size = L.size
                
class HoveringLabel(Label):
    background_color = ListProperty([0.3, 0.3, 0.3, 0.7])  # Default background color

    def __init__(self, **kwargs):
        super(HoveringLabel, self).__init__(**kwargs)
        # Ensure size and position updates redraw the background
        self.bind(pos=self.update_background, size=self.update_background)
        # Initial draw of the background
        self.draw_background()

    def draw_background(self):
        with self.canvas.before:
            self.bg_color = Color(*self.background_color)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

    def update_background(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        
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

class ActionButton(Button):
    inside_group = False
    icon = None
    def __init__(self, action_func, **kwargs):
        super().__init__(**kwargs)
        self.action_func = None
        if action_func is not None:
            self.set_action_func(action_func)
    def set_action_func(self, action_func):
        self.action_func = action_func
        self.bind(on_press = action_func)