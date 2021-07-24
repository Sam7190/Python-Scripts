# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:10:01 2021

@author: samir
"""

import time
import threading
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController


bury_shortcut = KeyCode(char='4')
preset = KeyCode(char='3')
mouse_move_delay = 0.8
bury_delay = 0.3
press = 0.05
bury_inv = 18

start_stop_key = KeyCode(char='1')
exit_key = KeyCode(char='2')


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
                # Get the bones from bank
                mouse.click(Button.left)
                time.sleep(mouse_move_delay)
                keyboard.press(Key.ctrl)
                keyboard.press(preset)
                time.sleep(press)
                keyboard.release(preset)
                keyboard.release(Key.ctrl)
                time.sleep(mouse_move_delay)
                
                # Bury bones
                start_time = time.time()
                current_time = time.time()
                while ((current_time - start_time) < bury_inv) and self.running:
                    keyboard.press(bury_shortcut)
                    time.sleep(press)
                    keyboard.release(bury_shortcut)
                    time.sleep(bury_delay)
                    current_time = time.time()
                    
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