# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 00:42:47 2021

@author: samir
"""

import time
import threading
from pynput.mouse import Button, Controller as mouseController
from pynput.keyboard import Listener, KeyCode, Key, Controller as keyboardController


smith_delay = 35.5
start_delay = 1
start_stop_key = KeyCode(char='1')
exit_key = KeyCode(char='2')


class ClickMouse(threading.Thread):
    def __init__(self, smith_delay, start_delay):
        super(ClickMouse, self).__init__()
        self.smith_delay = smith_delay
        self.start_delay = start_delay
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
                mouse.click(Button.left)
                time.sleep(self.start_delay)
                keyboard.press(Key.space)
                time.sleep(0.03)
                keyboard.release(Key.space)
                time.sleep(self.smith_delay)
            time.sleep(0.1)

mouse = mouseController()
keyboard = keyboardController()
click_thread = ClickMouse(smith_delay, start_delay)
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