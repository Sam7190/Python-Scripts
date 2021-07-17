# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:18:36 2021

@author: samir
"""

import time
import threading
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController


shortcut = KeyCode(char='5')
mouse_move_delay = 0.8
incense_inv_26 = 40
incense_inv_28 = 43
lag_factor = 1.5
lag_time = 3

mark_position_key = KeyCode(char='0')
pop_position_key = KeyCode(char='-')
start_stop_key = KeyCode(char='1')
exit_key = KeyCode(char='2')


class ClickMouse(threading.Thread):
    def __init__(self):
        super(ClickMouse, self).__init__()
        self.running = False
        self.program_running = True
        self.positions = [] # Pos 1: Banker, Pos 2: Logs, Pos 3: Withdraw
        self.is_first = True

    def start_clicking(self):
        if len(self.positions) == 3:
            self.running = True
        else:
            print("Did not complete positioning... ignoring start...")

    def stop_clicking(self):
        self.running = False

    def exit(self):
        self.stop_clicking()
        self.program_running = False

    def run(self):
        while self.program_running:
            while self.running:
                # Get the logs from bank
                mouse.position = self.positions[0] # Banker
                time.sleep(mouse_move_delay)
                mouse.click(Button.left)
                time.sleep(mouse_move_delay)
                mouse.position = self.positions[1] # Logs
                time.sleep(mouse_move_delay)
                mouse.click(Button.right)
                time.sleep(mouse_move_delay)
                mouse.position = self.positions[2] # Withdraw
                time.sleep(mouse_move_delay)
                mouse.click(Button.left)
                time.sleep(mouse_move_delay)
                keyboard.press(Key.esc)
                time.sleep(0.05)
                keyboard.release(Key.esc)
                time.sleep(mouse_move_delay*lag_factor)
                
                # Make incense (assumes incense is the default crafting target)
                keyboard.press(shortcut)
                time.sleep(0.05)
                keyboard.release(shortcut)
                time.sleep(mouse_move_delay*lag_factor) # Wait for pop-up
                keyboard.press(Key.space)
                time.sleep(0.05)
                keyboard.release(Key.space)
                time.sleep(incense_inv_28 if self.is_first else incense_inv_26)
                if self.is_first: self.is_first = False
                
                time.sleep(lag_time)
                    
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