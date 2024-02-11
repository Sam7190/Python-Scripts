# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:10:20 2024

@author: samir
"""
import socket_client
from functools import partial
import gameVariables as var
import numpy as np

quest_req = {(1, 3): 'self.playerTrack.player.actions == self.playerTrack.player.max_actions',
             (1, 8): "'fruit' in self.playerTrack.player.items",
             (2, 3): "self.quests[2, 6].status != 'started'",
             (2, 5): "self.quests[1, 6].status == 'complete'",
             (2, 6): "self.quests[2, 3].status != 'started'",
             (2, 7): "self.quests[1, 4].status == 'complete'",
             (3, 4): "self.playerTrack.player.Combat >= 40",
             (3, 6): "gqsts.StartofSkirmish()", # [INCOMPLETE] Figure out if beginning of skirmish for a city -- then choose that city
             (3, 8): "self.quests[1, 7].status == 'complete'",
             (4, 1): "self.quests[2, 5].status == 'complete'",
             (4, 7): "(self.quests[1, 2].status == 'complete') and (self.playerTrack.player.skills['Crafting'] >= 6)",
             (5, 5): "self.playerTrack.player.coins >= 40",
             (5, 6): "self.quests[1, 8].status == 'complete'",
             (5, 7): "self.quests[2, 8].status == 'complete'",
             (5, 8): "self.playerTrack.player.skills['Crafting'] >= 12"}

class gameQuests:
    def __init__(self, player, output, exitActionLoop, actionGrid, rbtwn):
        self.player = player
        self.output = output
        self.exitActionLoop = exitActionLoop
        self.actionGrid = actionGrid
        self.rbtwn = rbtwn

    def FindPet(self, _=None):
        P = self.player
        if P.paused:
            return
        excavating = P.activateSkill("Excavating")
        r = eval('rbtwn')(1, 2, None, excavating, 'Excavating ')
        if r <= excavating:
            P.useSkill("Excavating")
            eval('output')("You found the pet! You get 1 coin!", 'green')
            P.coins += 1
            P.PlayerTrack.Quest.update_quest_status((1, 1), 'complete')

        else:
            eval('output')("Unable to find the pet.", 'yellow')
        eval('exitActionLoop')()()

    def CleanHome(self, _=None):
        P = self.player
        if P.paused:
            return
        B = getQuest(1, 2)
        B.count = B.count + 1 if hasattr(B, 'count') else 1
        eval('output')(f"You spent an action cleaning the home (Total: {B.count})")
        if B.count >= 4:
            P.addItem('string', 1)
            P.addItem('beads', 1)
            P.PlayerTrack.Quest.update_quest_status((1, 2), 'complete')
        eval('exitActionLoop')()()

    def GaurdHome(self, _=None):
        P = self.player
        if P.paused:
            return
        B = getQuest(1, 3)
        def Reward():
            actions = {'Tin': eval('getItem')('tin', 1, action_amt=0),
                       'Iron': eval('getItem')('iron', 1, action_amt=0),
                       'Lead': eval('getItem')('lead', 1, action_amt=0),
                       'Copper': eval('getItem')('copper', 1, action_amt=0)}
            P.PlayerTrack.Quest.update_quest_status((1, 3), 'complete')
            eval('output')("The owner wants to reward you for gaurding the house! Choose one of the following:", 'blue')
            eval('actionGrid')(actions, False)
        def Consequence():
            P.PlayerTrack.Quest.update_quest_status((1, 3), 'failed')
        B.count = B.count + 1 if hasattr(B, 'count') else 1
        eval('output')("You spent an action gaurding the house")
        if B.count >= 2:
            eval('encounter')('Robber', [6, 6], [P.combatstyle], Reward, consequence=Consequence, background_img='images\\resized\\background\\cottage.png')
        else:
            eval('exitActionLoop')()()

    def OfferCraft(self, _=None):
        P = self.player
        if P.paused:
            return
        def Offer(space, _=None):
            sellprice = P.PlayerTrack.craftingTable.sell_value(space)
            persuasion = P.activateSkill("Persuasion")
            r = eval('rbtwn')(1, 4, None, sellprice+persuasion, 'Persuasion ')
            if r <= (sellprice + persuasion):
                def lvlUp(skill, _=None):
                    P.updateSkill(skill, 1, 6)
                    eval('exitActionLoop')()()
                P.activateSkill("Persuasion")
                P.PlayerTrack.craftingTable.rmv_craft(space)
                eval('output')("The boy accepted your craft!", 'green')
                P.PlayerTrack.Quest.update_quest_status((1, 4), 'complete')
                if (P.skills['Crafting'] < 6) or (P.skills['Persuasion'] < 6):
                    eval('output')("Choose a skill to level up:", 'blue')
                    actions = {skill: partial(lvlUp, skill) for skill in ['Crafting', 'Persuasion']}
                    eval('actionGrid')(actions, False)
                else:
                    eval('output')("Your Crafting and Persuasion are already level 6+", 'yellow')
                    eval('exitActionLoop')()()
            else:
                eval('output')("Unable to convince the boy to accept your craft.", 'yellow')
                eval('exitActionLoop')()()
        actions = {'Cancel':eval('exitActionLoop')(amt=0)}
        if P.PlayerTrack.craftingTable.space_items[0] > 1:
            actions["Craft 1"] = partial(Offer, 0)
        if P.PlayerTrack.craftingTable.space_items[1] > 1:
            actions["Craft 2"] = partial(Offer, 1)
        eval('actionGrid')(actions, False)

    def SpareWithBoy(self, _=None):
        P = self.player
        if P.paused:
            return
        attack = P.combat[P.attributes["Attack"]]
        r = eval('rbtwn')(1, 5, None, attack, 'Sparring ')
        if r <= attack:
            eval('output')("The young warrior is satisfied with your playful duel! Impact stability increases by 1.", 'green')
            P.stability_impact += 1
            P.PlayerTrack.Quest.update_quest_status((1, 5), 'complete')
        eval('exitActionLoop')()()

    def checkBook():
        P = self.player
        for item in P.items:
            if 'book' in item:
                return True
        return False

    def OfferBook(_=None):
        P = self.player
        if P.paused:
            return
        def Offer(book, _=None):
            P.addItem(book, -1)
            P.PlayerTrack.Quest.update_quest_status((1, 6), 'complete')
            eval('exitActionLoop')()()
        actions = {"Cancel": eval('exitActionLoop')(amt=0)}
        for item in P.items:
            if 'book' in item:
                actions[item] = partial(Offer, item)
        eval('actionGrid')(actions, False)

    def OfferSand(_=None):
        P = self.player
        if P.paused:
            return
        P.addItem('sand', -1)
        P.PlayerTrack.Quest.update_quest_status((1, 7), 'complete')
        eval('exitActionLoop')()()

    def ZooKeeper(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((1, 8), 'complete')
        eval('exitActionLoop')()()

    def ApplyCubes(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[2, 1], 'count'):
            P.PlayerTrack.Quest.quests[2, 1].count = 1
            eval('output')("Partially applied cubes")
        else:
            P.PlayerTrack.Quest.quests[2, 1].count += 1
            P.addItem('cooling cubes', -1)
            eval('output')("You applied the cubes successfully! The smith will now let you rent his facility for free!", 'green')
            P.free_smithing_rent = True
            P.PlayerTrack.Quest.update_quest_status((2, 1), 'complete')
        eval('exitActionLoop')()()

    def WaitforRobber(_=None):
        P = self.player
        if P.paused:
            return
        def Reward(_=None):
            eval('output')("The Robber is scared off! Market owners will now give you a discount of 1 coin for items costing 5+ coins!", 'green')
            P.city_discount = 1
            P.city_discount_threshold = 5
            P.PlayerTrack.Quest.update_quest_status((2, 2), 'complete')
        stealth = P.activateSkill("Stealth")
        r = eval('rbtwn')(0, 4, None, stealth, "Stealth ")
        if r <= stealth:
            P.useSkill("Stealth")
            eval('output')("You find and suprise the Robber!")
            eval('encounter')("Robber", [30, 30], [P.combatstyle], Reward, consequence=Reward, encounter=-1, background_img='images\\resized\\background\\city_night.png')
        else:
            eval('output')("You were not able to find the Robber", 'yellow')
            eval('exitActionLoop')()()

    def BeginProtection(_=None):
        P = self.player
        sorted_cities = sorted(var.cities)
        i = sorted_cities.index(P.birthcity)
        distances = np.concatenate((var.connectivity[:i, i], var.connectivity[i, i:]))
        # If there is a tie, pick a random furthest city
        furthest_city = np.random.choice(np.array(sorted_cities)[distances == np.max(distances)])
        eval('output')(f"The Nobleman requests you to protect him until reaching {furthest_city}", 'blue')
        P.PlayerTrack.Quest.quests[2, 3].furthest_city = furthest_city

    def MotherSerpent(_=None):
        P = self.player
        coord, distance, path = P.currenttile.findNearest('pond')
        eval('output')("The pond where the Mother Serpent lives is tinted green on the map!", 'blue')
        P.PlayerTrack.Quest.quests[2, 4].pond = coord
        P.parentBoard.gridtiles[coord].color = (1, 1.5, 1, 1)
        P.group["Companion"] = eval('npc_stats')(35)

    def LibrariansSecret(_=None):
        P = self.player
        coord, distance, path = P.currenttile.findNearest('oldlibrary')
        eval('output')("The old library where they librarian lost his book is tinted green on the map!", 'blue')
        P.PlayerTrack.Quest.quests[2, 5].coord = coord
        P.parentBoard.gridtiles[coord].color = (1, 1.5, 1, 1)
        P.PlayerTrack.Quest.quests[2, 5].has_book = False

    def HandOverBook(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((2, 5), 'complete')
        eval('output')("You can now gain 8xp from reading books!", 'green')
        P.standard_read_xp = 8

    def TheLetter(_=None):
        P = self.player
        sorted_cities = sorted(var.cities)
        i = sorted_cities.index(P.birthcity)
        distances = np.concatenate((var.connectivity[:i, i], var.connectivity[i, i:]))
        # If there is a tie, pick a random furthest city
        furthest_city = np.random.choice(np.array(sorted_cities)[distances == np.max(distances)])
        eval('output')(f"You are requested to take the letter to {furthest_city}", 'blue')
        P.PlayerTrack.Quest.quests[2, 6].furthest_city = furthest_city

    def GiveOresToSmith(_=None):
        P = self.player
        if P.paused:
            return
        def rewards(items, choices_left=2):
            actions = {(item[0].upper()+item[1:]):partial(choose_ore, item, items, choices_left) for item in items}
            eval('actionGrid')(actions, False)
        def choose_ore(item, from_items, choices_left, _=None):
            choices_left -= 1
            from_items.pop(item)
            P.addItem(item, 1)
            if choices_left > 0:
                rewards(from_items, choices_left)
            else:
                eval('exitActionLoop')()()
        for item in {'aluminum', 'nickel', 'tantalum', 'kevlium'}:
            P.addItem(item, -1)
        P.PlayerTrack.Quest.update_quest_status((2, 7), 'complete')
        eval('output')("Choose 2 of the following ores:", 'blue')
        rewards({'titanium', 'chromium', 'tungsten', 'diamond'}, 2)

    def PursuadeStealthMaster(_=None):
        P = self.player
        if P.paused:
            return
        if P.currenttile.tile not in {'enfeir', 'zinzibar'}:
            eval('output')("There is no steatlh master to persuade here!", 'yellow')
            eval('exitActionLoop')(amt=0)()
        elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'has_book') and P.PlayerTrack.Quest.quests[2, 8].has_book:
            eval('output')("You already own the book!", 'yellow')
            eval('exitActionLoop')(amt=0)()
        elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds') and (P.PlayerTrack.Quest.quests[2, 8]>0) and (P.currentcoord==P.PlayerTrack.Quest.quests[2, 8].coord):
            eval('output')(f"You need to wait {P.PlayerTrack.Quest.quests[2, 8].wait_rounds} more rounds for master to write the book", 'yellow')
            eval('exitActionLoop')(amt=0)()
        elif hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds') and (P.PlayerTrack.Quest.quests[2, 8]==0) and (P.currentcoord==P.PlayerTrack.Quest.quests[2, 8].coord):
            P.PlayerTrack.Quest.quests[2, 8].has_book = True
            eval('output')("The stealth master hands you the book, now present it to the quest giver at home!", 'blue')
            eval('exitActionLoop')('minor')()
        elif not hasattr(P.PlayerTrack.Quest.quests[2, 8], 'wait_rounds'):
            persuasion = P.activateSkill("Persuasion")
            r = eval('rbtwn')(1, 8, None, persuasion, 'Persuasion ')
            if r <= persuasion:
                eval('output')("You convinced the master stealth master to write you a special stealth book! Come back in 5 rounds to pick it up!", 'blue')
                P.PlayerTrack.Quest.quests[2, 8].wait_rounds = 5
                P.PlayerTrack.Quest.quests[2, 8].coord = P.currentcoord
            else:
                eval('output')("You fail to convince the master to write you a book", 'yellow')
            eval('exitActionLoop')()()
        else:
            # They could only get to this point if they go to the other city and persuade another master
            eval('output')("You have already persuaded a stealth master to write you a book!", 'red')
            eval('exitActionLoop')(amt=0)()

    def PresentStealthBook(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((2, 8), 'complete')
        P.updateSkill('Stealth', 2, 8)
        eval('exitActionLoop')('minor')()

    def FeedThePoor(_=None):
        P = self.player
        if P.paused:
            return
        def GiveItem(item, _=None):
            P.addItem(item, -1)
            P.PlayerTrack.Quest.quests[3, 1].food_given += 1
            eval('output')(f"You have distributed a total of {P.PlayerTrack.Quest.quests[3, 1].food_given} food!", 'blue')
            if P.PlayerTrack.Quest.quests[3, 1].food_given >= 10:
                P.PlayerTrack.Quest.update_quest_status((3, 1), 'complete')
                eval('output')("Your max eating per action increases by 2!", 'green')
                P.max_eating += 2
            eval('exitActionLoop')('minor')
        if not hasattr(P.PlayerTrack.Quest.quests[3, 1], 'food_given'):
            P.PlayerTrack.Quest.quests[3, 1].food_given = 0
        actions = {'Cancel': eval('exitActionLoop')(amt=0)}
        food_pieces = 0
        for item in P.items:
            if item in {'fruit', 'cooked meat', 'well cooked meat', 'cooked fish', 'well cooked fish'}:
                actions[item] = partial(GiveItem, item)
                food_pieces += 1
        if food_pieces == 0:
            eval('output')("You do not have any cooked food or fruit to distribute!", 'yellow')
            eval('exitActionLoop')(amt=0)()
        else:
            eval('actionGrid')(actions, False)

    def resetFitnessTraining(_=None):
        P = self.player
        eval('output')("The three adventurers leave you and you must restart the Fitness Training", 'red')
        for i in range(3):
            P.group.pop(f'Adventurer {i+1}')
        P.PlayerTrack.Quest.update_quest_status((3, 2), 'not started')

    def finishFitnessTraining(_=None):
        P = self.player
        eval('output')("The three adventurers leave you")
        for i in range(3):
            P.group.pop(f'Adventurer {i+1}')
        P.PlayerTrack.Quest.update_quest_status((3, 2), 'complete')
        for skill in ['Gathering', 'Excavating', 'Survival']:
            P.updateSkill(skill, 1, 8)
        eval('output')("Max fatigue increases by 3", 'green')
        P.max_fatigue += 3

    def FitnessTraining(_=None):
        P = self.player
        eval('output')("3 Adventurers group up with you. First go to any mountain and find ores worth a total of 5 coins (sell cost) and climb to the top.", 'blue')
        for i in range(3):
            P.group[f'Adventurer {i+1}'] = eval('npc_stats')(15)
        P.PlayerTrack.Quest.quests[3, 2].total_ore_cost = 0
        P.PlayerTrack.Quest.quests[3, 2].reached_top = False
        P.PlayerTrack.Quest.quests[3, 2].completed_mountain = False
        P.PlayerTrack.Quest.quests[3, 2].meat_collected = 0

    def LearnFromMonk(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.quests[3, 3].convinced_monk += 1
        if P.PlayerTrack.Quest.quests[3, 3].convinced_monk >= 3:
            P.PlayerTrack.Quest.update_quest_status((3, 3), 'completed')
            for skill in ['Excavating', 'Survival']:
                P.updateSkill(skill, 1, 12)
        eval('exitActionLoop')()()

    def SearchForMonster(_=None):
        P = self.player
        if P.paused:
            return
        excavating = P.activateSkill("Excavating")
        r = eval('rbtwn')(1, 12, None, excavating, 'Excavating ')
        if r <= excavating:
            def monster_flees(rewarded=True, _=None):
                P.PlayerTrack.Quest.update_quest_status((3, 4), 'completed')
                if rewarded:
                    P.updateAttribute("Cunning", 1, 8)
                    P.updateAttribute("Technique", 1, 8)
            eval('encounter')('Monster', [55, 55], ['Physical', 'Elemental', 'Trooper', 'Wizard'], monster_flees, consequence=partial(monster_flees, False), background_img='images\\resized\\background\\city_night.png')
        else:
            eval('output')("You fail to find the monster.", 'yellow')
            eval('exitActionLoop')()()

    def TeachFishing(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[3, 5], 'count'): P.PlayerTrack.Quest.quests[3, 5].count = 0
        gathering = P.activateSkill("Gathering")
        r = eval('rbtwn')(1, 8, None, gathering, 'Gathering ')
        if r <= gathering:
            P.useSkill("Gathering")
            P.PlayerTrack.Quest.quests[3, 5].count += 1
            eval('output')(f'Success! So far, taught {P.PlayerTrack.Quest.quests[3, 5].count} men to fish', 'blue')
            if P.PlayerTrack.Quest.quests[3, 5].count >= 6:
                P.PlayerTrack.Quest.update_quest_status((3, 5), 'complete')
                P.coins += 8
                P.updateSkill('Gathering', 1, 8)
        else:
            eval('output')("Unable to teach a man to fish.", 'yellow')
        eval('exitActionLoop')()()

    def StartofSkirmish(_=None):
        P = self.player
        firstAction = P.actions == P.max_actions
        sufficientMinor = P.minor_actions >= 2
        skirmStart = False
        for sk in eval('Skirmishes')[0]:
            if (P.birthcity in sk) and (eval('Skirmishes')[0][sk] == 2):
                skirmStart = True
        return firstAction * sufficientMinor * skirmStart

    def JoinFight(_=None):
        P = self.player
        city_against = P.PlayerTrack.Quest.quests[3, 6].city_against
        foename = city_against[0].upper()+city_against[1:]+' Fighter'
        foestyle = var.cities[city_against]['Combat Style']
        eval('encounter')(foename, [30, 50], [foestyle], {'coins':[3, 6]}, enc=0)

    def TransportToField(_=None):
        P = self.player
        skirmishesPossible = set()
        for sk in eval('Skirmishes')[0]:
            if (P.birthcity in sk) and (eval('Skirmishes')[0][sk] == 2):
                skirmishesPossible = skirmishesPossible.union(sk.difference({P.birthcity}))
        skirmishesPossible = list(skirmishesPossible)
        city_against = np.random.choice(skirmishesPossible) if len(skirmishesPossible) > 0 else skirmishesPossible[0]
        coord, distance, path = P.currenttile.findNearest(city_against)
        mid_i = int(np.median(np.arange(len(path))))
        mid_tile = path[mid_i]
        P.moveto((mid_tile.gridx, mid_tile.gridy), False, True)
        P.PlayerTrack.Quest.quests[3, 6].city_against = city_against
        P.PlayerTrack.Quest.quests[3, 6].action = 1
        JoinFight()

    def PresentCraftBooks(_=None):
        P = self.player
        if P.paused:
            return
        P.addItem("crafting book", -3)
        P.PlayerTrack.Quest.update_quest_status((3, 7), 'complete')
        if P.birthcity == 'fodker':
            P.addItem('old fodker cloth', 3)
        else:
            eval('output')(f"Village eval('output') efficiency increased by 1 for {P.birthcity}!", 'green')
            var.capital_info[P.birthcity]['efficiency'] += 1
            socket_client.send('[EFFICIENCY]', P.birthcity)
        eval('exitActionLoop')()()

    def DistributeSand(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[3, 8], 'count'):
            P.PlayerTrack.Quest.quests[3, 8].count = 1
        else:
            P.PlayerTrack.Quest.quests[3, 8].count += 1
        eval('output')(f"Delivered {P.PlayerTrack.Quest.quests[3, 8].count} bags of sand so far", 'blue')
        P.addItems('sand', -1)
        if P.PlayerTrack.Quest.quests[3, 8].count >= 10:
            P.PlayerTrack.Quest.update_quest_status((3, 8), 'complete')
            eval('output')("New glass making school gives you 3 pieces of glass", 'green')
            P.addItem('glass', 3)
        eval('exitActionLoop')('minor')()

    def FillLibrary(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 1], 'books_given'):
            P.PlayerTrack.Quest.quests[4, 1].books_given = set()
        def give_book(book, _=None):
            P.PlayerTrack.Quest.quests[4, 1].books_given.add(book)
            eval('output')("You have now given: {', '.join(list(P.PlayerTrack.Quest.quests[4, 1].books_given))}", 'blue')
            if len(P.PlayerTrack.Quest.quests[4, 1].books_given) >= 5:
                P.PlayerTrack.Quest.update_quest_status((4, 1), 'complete')
                P.updateSkill("Critical Thinking", 2 if P.skills['Critical Thinking'] < 8 else 1)
            eval('exitActionLoop')('minor')()
        actions = {'Cancel':eval('exitActionLoop')(amt=0)}
        for item in P.items:
            if ('book' in item) and (item not in P.PlayerTrack.Quest.quests[4, 1].books_given):
                actions[item] = partial(give_book, item)
        eval('actionGrid')(actions, False)

    def FindWarrior(_=None):
        P = self.player
        if P.paused:
            return
        city = P.currenttile.tile
        if city not in var.cities:
            eval('output')("You are not in a city!", 'yellow')
            return
        elif city == P.birthcity:
            eval('output')("You need to search in cities part from yours!", 'yellow')
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 2], 'cities_searched'):
            P.PlayerTrack.Quest.quests[4, 2].cities_searched = set()
            P.PlayerTrack.Quest.quests[4, 2].cities_found = set()
        if city not in P.PlayerTrack.Quest.quests[4, 2].cities_searched:
            P.PlayerTrack.Quest.quests[4, 2].cities_searched.add(city)
            excavating = P.activateSkill("Excavating")
            r = eval('rbtwn')(1, 12, None, excavating, 'Excavating ')
            if r <= excavating:
                P.useSkill("Excavating")
                eval('output')(f"You found a {P.birthcity} warrior in {city}! He responds to your call and heads to {P.birthcity}!", 'green')
                P.PlayerTrack.Quest.quests[4, 2].cities_found.add(city)
                if len(P.PlayerTrack.Quest.quests[4, 2].cities_found) >= 7:
                    P.PlayerTrack.Quest.update_quest_status((4, 2), 'complete')
                    eval('output')("Your Hit Points and Stability are boosted by 2!", 'green')
                    P.boosts[P.attributes["Hit Points"]] += 2
                    P.current[P.attributes["Hit Points"]] += 2
                    P.boosts[P.attributes["Stability"]] += 2
                    P.boosts[P.attributes["Stability"]] += 2
            elif (len(P.PlayerTrack.Quest.quests[4, 2].cities_searched) - len(P.PlayerTrack.Quest.quests[4, 2].cities_found)) >= 7:
                eval('output')("It is no longer possible for you to find 7 warriors!", 'red')
                P.PlayerTrack.Quest.update_quest_status((4, 2), 'failed')
            else:
                failsleft = 7 - (len(P.PlayerTrack.Quest.quests[4, 2].cities_searched) - len(P.PlayerTrack.Quest.quests[4, 2].cities_found))
                eval('output')(f"You failed to find a warrior in {city}! You can only fail {failsleft} more times!", 'red')
            eval('exitActionLoop')()()
        else:
            eval('output')("You have already searched this city!", 'yellow')

    def ConvinceMarketLeader(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 3], 'count'):
            P.PlayerTrack.Quest.quests[4, 3].count = 0
        bartering = P.activateSkill("Bartering")
        persuasion = P.activateSkill("Persuasion")
        r = eval('rbtwn')(1, 24, None, bartering+persuasion, 'Convincing Market Leader ')
        if r <= (bartering + persuasion):
            P.useSkill("Bartering")
            P.useSkill("Persuasion")
            eval('output')("You convinced the market leader to lower his prices!", 'green')
            P.PlayerTrack.Quest.quests[4, 3].count += 1
            if P.PlayerTrack.Quest.quests[4, 3].count >= 6:
                P.PlayerTrack.Quest.update_quest_status((4, 3), 'complete')
                eval('output')(f"Market prices reduced by 1 (min=1) in {P.birthcity}!", 'green')
                var.capital_info[P.birthcity]['discount'] += 1
                socket_client.send('[DISCOUNT]', P.birthcity)
            else:
                eval('output')("You still need to convince {6 - P.PlayerTrack.Quest.quests[4, 3].count} more market leaders.", 'blue')
        else:
            eval('output')("You were unable to convince them this time.", 'yellow')
        eval('exitActionLoop')()()

    def PursuadeTrader(_=None):
        P = self.player
        if P.paused:
            return
        persuasion = P.activateSkill("Persuasion")
        r = eval('rbtwn')(1, 12, None, persuasion, "Persuasion ")
        if r <= persuasion:
            P.useSkill("Persuasion")
            eval('output')(f"Successfully pursuaded trader to start coming to {P.birthcity}!", 'green')
            P.PlayerTrack.Quest.quests[4, 4].count += 1
            if P.PlayerTrack.Quest.quests[4, 4].count >= 3:
                P.PlayerTrack.Quest.update_quest_status((4, 4), 'complete')
                eval('output')(f"Traders now have a 1/8 chance of appearing in {P.birthcity}!", 'green')
                var.capital_info[P.birthcity]['trader allowed'] = True
                socket_client.send('[TRADER ALLOWED]', P.birthcity)
        else:
            eval('output')("Unable to convince the trader.", 'yellow')
        eval('exitActionLoop')()()

    def FindAndPursuadeLeader(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 5], 'count'):
            P.PlayerTrack.Quest.quests[4, 5].count = 0
        if not hasattr(P.PlayerTrack.Quest.quests[4, 5], 'skirmish'):
            P.PlayerTrack.Quest.quests[4, 5].skirmish = {P.currenttile.tile, P.birthcity}
        if {P.currenttile.tile, P.birthcity} != P.PlayerTrack.Quest.quests[4, 5].skirmish:
            eval('output')(f"You already began trying to reduce tension between {' and '.join(list(P.PlayerTrack.Quest.quests[4, 5].skirmish))}. You must stick to that!", 'yellow')
            return
        excavating = P.activateSkill("Excavating")
        r = eval('rbtwn')(1, 10, None, excavating, "Excavating ")
        if r <= excavating:
            P.useSkill("Excavating")
            eval('output')("You found a leader!")
            persuasion = P.activateSkill("Persuasion")
            r = eval('rbtwn')(1, 12, None, persuasion, "Persuasion ")
            if r <= persuasion:
                P.useSkill("Persuasion")
                P.PlayerTrack.Quest.quests[4, 5].count += 1
                eval('output')(f"You convinced the leader to lessen the war effort! {5 - P.PlayerTrack.Quest.quests[4, 5].count} left to go!", 'blue')
                if P.PlayerTrack.Quest.quests[4, 5].count >= 5:
                    P.PlayerTrack.Quest.update_quest_status((4, 5), 'complete')
                    skirmish = frozenset([P.birthcity, P.currenttile.tile])
                    eval('output')("You reduced the tensions between {' and '.join(list(skirmish))} by a factor of 3!", 'green')
                    eval('Skirmishes')[1][skirmish] += 3
                    socket_client.send('[REDUCED TENSION]', [skirmish, 3])
            else:
                eval('output')("Failed to persuade the leader.", 'yellow')
        else:
            eval('output')("Failed to find a leader.", 'yellow')
        eval('exitActionLoop')()()

    def StoreTatteredBook(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 6], 'count'):
            P.PlayerTrack.Quest.quests[4, 6].count = 1
        else:
            P.PlayerTrack.Quest.quests[4, 6].count += 1
        eval('output')(f"You found an old book containing long lost history! You have found {P.PlayerTrack.Quest.quests[4, 6].count} so far.", 'blue')
        eval('exitActionLoop')()()

    def DeliverTatteredBooks(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((4, 6), 'complete')
        coins = min([40, 10*P.PlayerTrack.Quest.quests[4, 6].count])
        eval('output')(f"You received {coins} coins!", 'green')
        P.coins += coins
        eval('exitActionLoop')()()

    def IncreaseCapacity(city, amt, _=None):
        P = self.player
        var.capital_info[city]['capacity'] += amt
        if P.home[city]:
            P.max_capacity += amt

    def HomeRenovations(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 7], 'remaining'):
            P.PlayerTrack.Quest.quests[4, 7].remaining = {'bark':5, 'clay':3, 'glass':3, 'leather':2}
        def distribute(item, _=None):
            P.PlayerTrack.Quest.quests[4, 7].remaining[item] -= 1
            allZero = True
            for val in P.PlayerTrack.Quest.quests[4, 7].remaining.values():
                if val > 0:
                    allZero = False
            if allZero:
                P.PlayerTrack.Quest.update_quest_status((4, 7), 'complete')
                eval('output')(f"Homes in {P.birthcity} have increased capacity of 3!", 'green')
                IncreaseCapacity(P.birthcity, 3)
                socket_client.send('[CAPACITY]', [P.birthcity, 3])
            eval('exitActionLoop')('minor')()
        actions = {'Cancel':eval('exitActionLoop')(amt=0)}
        for item in P.items:
            if (item in P.PlayerTrack.Quest.quests[4, 7].remaining) and (P.PlayerTrack.Quest.quests[4, 7].remaining[item] > 0):
                actions[item[0].upper()+item[1:]] = partial(distribute, item)
        eval('actionGrid')(actions, False)

    def ConvinceWarriorsToRaid(_=None):
        P = self.player
        if P.paused:
            return
        if not hasattr(P.PlayerTrack.Quest.quests[4, 8], 'cities'):
            P.PlayerTrack.Quest.quests[4, 8].cities = {}
        elif (len(P.PlayerTrack.Quest.quests[4, 8]) >= 4) and (P.currenttile.tile not in P.PlayerTrack.Quest.quests[4, 8].cities):
            eval('output')("You have already started looking in other cities!", 'yellow')
            return
        if (P.birthcity not in P.PlayerTrack.Quest.quests[4, 8].cities) and (P.currenttile.tile != P.birthcity):
            eval('output')("You need to start with your city first!", 'yellow')
            return
        elif (P.currenttile.tile in P.PlayerTrack.Quest.quests[4, 8].cities) and (P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] >= 3):
            eval('output')("You have already found and convinced 3 warriors in this city", 'yellow')
            return
        elif P.currenttile.tile in eval('Skirmishes')[2][P.birthcity]:
            eval('output')("You cannot convince any warriors in this city due to tensions!", 'yellow')
            return
        elif P.currenttile.tile not in P.PlayerTrack.Quest.quests[4, 8].cities:
            # Initiate the count for the city
            P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] = 0
        excavating = P.activateSkill("Excavating")
        r = eval('rbtwn')(1, 12, None, excavating, 'Excavating ')
        if r <= excavating:
            eval('output')("You found a warrior")
            persuasion = P.activateSkill("Persuasion")
            r = eval('rbtwn')(1, 10, None, persuasion, 'Persuasion ')
            if r <= persuasion:
                P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] += 1
                if P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile] >= 3:
                    eval('output')(f"You have convinced 3 warriors in {P.currenttile.tile} to raid the caves. They set out and do so!", 'green')
                    if len(P.PlayerTrack.Quest.quests[4, 8].cities) >= 4:
                        allFinished = True
                        for val in P.PlayerTrack.Quest.quests[4, 8].cities.values():
                            if val < 3:
                                allFinished = False
                        if allFinished:
                            P.PlayerTrack.Quest.update_quest_status((4, 8), 'completed')
                else:
                    eval('output')("You persuade them to raid the caves once all three in the {P.currentile.tile} are ready. You have convinced {P.PlayerTrack.Quest.quests[4, 8].cities[P.currenttile.tile]} so far in {P.currentile.tile}!", 'blue')
            else:
                eval('output')("Unable to persuade the warrior", 'yellow')
        else:
            eval('output')("Unable to find any warrior", 'yellow')
        eval('exitActionLoop')()()

    def ConvinceWarriorToJoin(_=None):
        P = self.player
        if P.paused:
            return
        # Assumption: Logic to determine whether finding warrior in this city is valid is handled somewhere else
        excavating = P.activateSkill("Excavating")
        r = eval('rbtwn')(1, 12, None, excavating, 'Excavating ')
        if r <= excavating:
            lvl = max(eval('rbtwn')(40, 80, max([1, excavating//3])))
            eval('output')(f"You found a warrior lvl {lvl}")
            persuasion = P.activateSkill("Persuasion")
            r = eval('rbtwn')(1, 6 + round((80-lvl)/6), None, persuasion, 'Persuasion ')
            if r <= persuasion:
                eval('output')("You convinced the warrior to join you for 6 rounds! Any other warriors leave you group.", 'green')
                P.has_warrior = 6
                warriorstats = eval('npc_stats')(lvl)
                P.group["Warrior"] = warriorstats
            else:
                eval('output')("Unable to convince them to join you.", 'yellow')
        else:
            eval('output')("Unable to find any warriors.", 'yellow')
        eval('exitActionLoop')()()

    def GoldStart(_=None):
        P = self.player
        P.PlayerTrack.Quest.quests[5, 1].has_gold = False

    def SmithGold(_=None):
        P = self.player
        if P.paused:
            return
        persuasion = P.activateSkill("Persuasion")
        r = eval('rbtwn')(1, 14, None, persuasion, 'Persuasion ')
        if r <= persuasion:
            eval('output')("You were able to convince the smith to make the dagger! Mayor rewards you with 20 coins and your impact stability reduces by 1.", 'green')
            P.PlayerTrack.Quest.update_quest_status((5, 1), 'complete')
            P.coins += 20
            P.stability_impact += 1
        else:
            eval('output')("Unable to convince the smith", 'yellow')
        eval('exitActionLoop')()()

    def PlaceFriend(_=None):
        P = self.player
        allPos = []
        for tileType in ['randoms', 'ruins', 'battle1', 'battle2', 'wilderness']:
            allPos += var.positions[tileType]
        randPos = np.random.choice(allPos)
        P.PlayerTrack.Quest.quests[5, 2].coord = randPos
        P.PlayerTrack.Quest.quests[5, 2].has_friend = False

    def ShowFriend(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((5, 2), 'complete')
        eval('output')("You gain 20 coins and max actions per round increases by 1!", 'green')
        P.coins += 20
        P.max_actions += 1
        eval('exitActionLoop')('minor')()

    def KilledMonster(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((5, 3), 'complete')
        P.coins += 30
        for city in eval('Skirmishes')[2][P.birthcity]:
            skirmish = frozenset([city, P.birthcity])
            eval('Skirmishes')[1][skirmish] += 1
            eval('output')("You reduced the tensions between {' and '.join(list(skirmish))} by a factor of 1!", 'green')
            socket_client.send('[REDUCED TENSION]', [skirmish, 1])
        eval('exitActionLoop')('minor')()

    def ShowRareOre(_=None):
        P = self.player
        if P.paused:
            return
        rares = {'shinopsis', 'ebony', 'astatine', 'promethium'}
        raresLeft = rares.intersection(P.items)
        def giveRare(raresSelected, raresLeft, _=None):
            if len(raresSelected) >= 2:
                P.PlayerTrack.Quest.update_quest_status((5, 4), 'complete')
                for rare in raresSelected:
                    atr = f'Def-{var.ore_properties[rare][0]}'
                    P.boosts[P.attributes[atr]] += 2
                    P.current[P.attributes[atr]] += 2
                    eval('output')(f"{atr} Boosted by 2!", 'green')
                eval('exitActionLoop')()()
            else:
                actions = {'Cancel': eval('exitActionLoop')(amt=0)}
                for rare in raresLeft:
                    possibleSelect = raresSelected.union({rare})
                    possibleLeft = raresLeft.difference({rare})
                    actions[rare] = partial(giveRare, possibleSelect, possibleLeft)
                eval('actionGrid')(actions, False)
        giveRare(set(), raresLeft)

    def ConvinceBarterMaster(_=None):
        P = self.player
        if P.paused:
            return
        if P.coins < 40:
            eval('output')("You lack the ability to pay him!", 'yellow')
            return
        persuasion = P.activateSkill("Persuasion")
        r = eval('rbtwn')(1, 20, None, persuasion, 'Persuasion ')
        if r <= persuasion:
            P.useSkill("Persuasion", 2, 7)
            eval('output')("You convinced the Bartering Master to teach the mayor his knowledge! You pay him 40 coins. Your max minor actions increase by 4!", 'green')
            P.coins -= 40
            P.PlayerTrack.Quest.update_quest_status((5, 5), 'complete')
            P.max_minor_actions += 4
        else:
            eval('output')("Unable to convince the Bartering Master", 'yellow')
        eval('exitActionLoop')()()

    def HighlightMammothTile(_=None):
        P = self.player
        P.parentBoard.gridtiles[P.PlayerTrack.Quest.quests[1, 8].coord_found].color = (1, 1.5, 1, 1)
        eval('output')("In case you forgot, the tile you found the baby mammoth is tinted green")

    def KilledMammoth(_=None):
        P = self.player
        if P.paused:
            return
        P.PlayerTrack.Quest.update_quest_status((5, 6), 'complete')
        eval('output')("You are awarded 3 gems and 3 diamonds!", 'green')
        P.addItem('gems', 3)
        P.addItem('diamond', 3)
        eval('exitActionLoop')('minor')()

    def AskMasterStealthForMayor(_=None):
        P = self.player
        if P.paused:
            return
        persuasion = P.activateSkill("Persuasion")
        cunning = P.current[P.attributes["Cunning"]]
        critthink = P.current[P.attributes["Critical Thinking"]]
        r = eval('rbtwn')(1, 36, None, (persuasion+cunning+critthink), 'Convicing ')
        if r <= (persuasion + cunning + critthink):
            eval('output')("You convinced the Stealth Master to teach the mayor! You get 2 stealth books and 1 free lesson to use.", 'green')
            P.PlayerTrack.Quest.update_quest_status((5, 7), 'completed')
            P.addItem("stealth book", 2)
            P.PlayerTrack.Quest.quests[5, 7].used_lesson = False
        else:
            eval('output')("You failed to convince the master.", 'yellow')
            if not hasattr(P.PlayerTrack.Quest.quests[5, 7], 'count'):
                P.PlayerTrack.Quest.quests[5, 7].count = 1
            else:
                P.PlayerTrack.Quest.quests[5, 7].count += 1
                if P.PlayerTrack.Quest.quests[5, 7].count >= 2:
                    P.PlayerTrack.Quest.update_quest_status((5, 7), 'failed')
        eval('exitActionLoop')()()

    def GiftPerfectCraft(_=None):
        P = self.player
        if P.paused:
            return
        perfectCraft = {'gems', 'rubber', 'glass', 'ceramic', 'leather', 'scales', 'beads'}
        gifting = False
        def giftCraft(space):
            P.PlayerTrack.craftingTable.rmv_craft(space)
            P.PlayerTrack.Quest.update_quest_status((5, 8), 'completed')
            eval('output')(f"You can now train in {P.birthcity} for free with Adept trainers and half price at Master trainers!", 'green')
            P.training_discount = True
        for space in [0, 1]:
            if P.PlayerTrack.craftingTable.getItems(space) == perfectCraft:
                giftCraft(space)
                gifting = True
                break
        if not gifting:
            eval('output')(f"You do not posses the perfect craft of: {', '.join(list(perfectCraft))}", 'yellow')
        eval('exitActionLoop')('minor')()
    
# Order: Action Name, Action Condition, Action Function
