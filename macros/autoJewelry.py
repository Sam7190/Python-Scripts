# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 14:39:45 2021

@author: samir
"""

import time
import threading
import numpy as np
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController

preset = Key.f4
delay = 0.8
press = 0.05
run = 6.5
inv = 50.5

mark_position_key = KeyCode(char='0')
pop_position_key = KeyCode(char='-')
start_stop_key = KeyCode(char='`')
exit_key = KeyCode(char='2')

def random_delay(delay_speed, proportion):
    return delay_speed + np.random.rand()*(delay_speed * proportion)

def random_sleep(delay_speed, proportion=0.3):
    time.sleep(random_delay(delay_speed, proportion))
    
def quick_press(key, press=press):
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
        
def click_mouse(position, side=Button.left, delay=delay):
    mouse.position = position
    random_sleep(delay)
    mouse.click(side)

class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True
        self.positions = [] # Pos 1: Furnace, Pos 2: Bank

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                # Assumes already in Bank
                quick_press(preset)
                random_sleep(delay)
                click_mouse(self.positions[0])
                random_sleep(run)
                quick_press(Key.space)
                random_sleep(inv)
                click_mouse(self.positions[1])
                random_sleep(run)
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
    elif (key == mark_position_key) and not click_thread.running:
        if len(click_thread.positions) < 2:
            click_thread.positions.append(mouse.position)
    elif (key == pop_position_key) and not click_thread.running:
        if len(click_thread.positions) > 0:
            click_thread.positions.pop()


with Listener(on_press=on_press) as listener:
    listener.join()