# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 07:09:36 2024

@author: samir
"""

hallmarks = {'benfriege': 
{'hallmark': 'Grand Library',
 'description': 
"""At this library you can read books or check them out.

  - Reading Books: From a choice of three random books, read one (minor action),
                   however, you get fatigue equal to your reading fatigue.
                   To clear your reading fatigue back to 0, you must go four
                   actions without reading (including knowledge books).
        > Skill Lvl 0-3: 1xp gauranteed
        > Skill Lvl 4-5: 1xp @ 67% probability
        > Skill Lvl 6-7: 1xp @ 33% probability
                   
  - Checking Out Books: From a choice of three random books, checkout one
                        (minor action) only a trainer can teach you the book. 
                        Go to an adept or master trainer of that skill to learn.
        > 2xp gauranteed
        > Skill Lvl 0-3: trainer charges 1 coin
        > Skill Lvl 4-5: trainer charges 2 coins
        > Skill Lvl 6-7: trainer charges 3 coins"""