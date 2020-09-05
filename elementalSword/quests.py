# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 22:08:54 2020

@author: samir
"""

def FindPet(_=None):
    P = lclPlayer()
    if P.paused:
        return
    excavating = P.activateSkill("Excavating")
    r = rbtwn(1, 3, None, excavating, 'Excavating ')
    if r <= excavating:
        P.useSkill("Excavating")
        output("You found the pet! You get 1 coin!", 'green')
        P.PlayerTrack.Quest.update_quest_status((1, 1), 'complete')
        P.coins += 1
    else:
        output("Unable to find the pet.", 'yellow')
    exitActionLoop()()

# Order: Action Name, Action Condition, Action Function
city_actions = {(1, 1): ["Find Pet", "True", FindPet]}