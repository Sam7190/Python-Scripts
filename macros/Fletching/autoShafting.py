# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:37:39 2021

@author: samir
"""

import time
import threading
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController


shortcut = KeyCode(char='4')
delay = 0.8
press = 0.05
inv = 49.5

mark_position_key = KeyCode(char='0')
pop_position_key = KeyCode(char='-')
start_stop_key = KeyCode(char='1')
exit_key = KeyCode(char='2')


class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True
        self.positions = [] # Pos 1: Banker, Pos 2: Bones, Pos 3: Withdraw

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
                mouse.position = self.positions[0] # Banker
                time.sleep(delay)
                mouse.click(Button.left)
                time.sleep(delay)
                mouse.position = self.positions[1] # Logs
                time.sleep(delay)
                mouse.click(Button.right)
                time.sleep(delay)
                mouse.position = self.positions[2] # Withdraw
                time.sleep(delay)
                mouse.click(Button.left)
                time.sleep(delay)
                keyboard.press(Key.esc)
                time.sleep(press)
                keyboard.release(Key.esc)
                time.sleep(delay)
                
                # Cut Logs
                keyboard.press(shortcut)
                time.sleep(press)
                keyboard.release(shortcut)
                time.sleep(delay)
                keyboard.press(Key.space)
                time.sleep(inv)
                    
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
        if len(click_thread.positions) < 3:
            click_thread.positions.append(mouse.position)
    elif (key == pop_position_key) and not click_thread.running:
        if len(click_thread.positions) > 0:
            click_thread.positions.pop()

with Listener(on_press=on_press) as listener:
    listener.join()