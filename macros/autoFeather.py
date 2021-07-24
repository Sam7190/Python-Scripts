# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:48:05 2021

@author: samir
"""

import time
import threading
import numpy as np
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController

shortcut = KeyCode(char='4')
delay = 0.8
press = 0.05
# inventory speed: 71 for feathering, 33 for tipping
inv = 33

start_stop_key = KeyCode(char='1')
exit_key = KeyCode(char='2')

def random_delay(delay_speed, proportion):
    return delay_speed + np.random.rand()*(delay_speed * proportion)

def random_sleep(delay_speed, proportion=0.3):
    time.sleep(random_delay(delay_speed, proportion))

class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True

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
                keyboard.press(shortcut)
                random_sleep(press)
                keyboard.release(shortcut)
                random_sleep(delay)
                keyboard.press(Key.space)
                random_sleep(press)
                keyboard.release(Key.space)
                random_sleep(inv, 0.04)
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


with Listener(on_press=on_press) as listener:
    listener.join()