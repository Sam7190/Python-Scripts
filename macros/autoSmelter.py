# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:30:28 2021

@author: samir
"""

import time
import threading
import numpy as np
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController

total_pos = 0
delay = 0.8
press = 0.05
inv = 50.5
deposit_key = KeyCode(char='1')

mark_position_key = KeyCode(char='0')
pop_position_key = KeyCode(char='-')
start_stop_key = KeyCode(char='`')
exit_key = KeyCode(char='2')

def random_delay(delay_speed, proportion):
    return delay_speed + np.random.rand()*(delay_speed * proportion)

def random_sleep(delay_speed, proportion=0.3):
    time.sleep(random_delay(delay_speed, proportion))
        
def quick_press(key, press=press, delay=delay, end_sleep=delay):
    if type(key) is list:
        for k in key:
            keyboard.press(k)
    else:
        keyboard.press(key)
    random_sleep(press)
    if type(key) is list:
        for k in key:
            keyboard.release(k)
    else:
        keyboard.release(key)
    if end_sleep: random_sleep(end_sleep)
        
def click_mouse(position, side=Button.left, delay=delay, end_sleep=delay):
    mouse.position = position
    random_sleep(delay)
    mouse.click(side)
    if end_sleep: random_sleep(end_sleep)

class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True
        self.positions = [] # Pos 1: Furnace, Pos 2: Bank, Pos 3: Inv slot
        self.cur_pos = -1

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False
        
    def click_mouse(self, side=Button.left, delay=delay, end_sleep=delay):
        self.cur_pos += 1
        if self.cur_pos >= len(self.positions):
            self.cur_pos = 0
        click_mouse(self.positions[self.cur_pos], side, delay, end_sleep)

    def run(self):
        while self.program_running:
            while self.running:
                # Assumes already at Bank
                mouse.click(Button.left)
                random_sleep(delay)
                quick_press(deposit_key)
                quick_press(Key.space, end_sleep=inv)
            time.sleep(0.1)

mouse = mouseController()
keyboard = keyboardController()
click_thread = ClickMouse()
click_thread.start()

def on_press(key):
    if key == start_stop_key:
        if click_thread.running:
            click_thread.stop_clicking()
        else:
            click_thread.start_clicking()
    elif key == exit_key:
        click_thread.exit()
        listener.stop()
    elif (key == mark_position_key) and (total_pos > 0) and (not click_thread.running):
        if len(click_thread.positions) < total_pos:
            click_thread.positions.append(mouse.position)
    elif (key == pop_position_key) and (total_pos > 0) and (not click_thread.running):
        if len(click_thread.positions) > 0:
            click_thread.positions.pop()


with Listener(on_press=on_press) as listener:
    listener.join()