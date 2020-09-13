import numpy as np
from copy import deepcopy as dc
from inspect import currentframe, getframeinfo

agility = 2
cunning = 0
technique = 0
hitpoints = 2
attack = 1
stability = 0
defphys = 0
defwiz = 0
defelem = 0
deftroop = 0
attackstyle = 'p'

Round = [0]

fatigue = [0, None]
defpos = {'p':6,'w':7,'e':8,'t':9}
combat = [agility, cunning, technique, hitpoints, attack, stability, defphys, defwiz, defelem, deftroop, attackstyle]
default = ['Agility', 'Cunning', 'Technique', 'Hit Points', 'Attack', 'Stability', 'Def - Physical', 'Def - Wizard', 'Def - Elemental', 'Def - Trooper']
acronym = ['a','c','t','h','atk','s','dp','dw','de','dt']

origBase = {'Tamarania,Demetry':3,
            'Tamarania,Enfeir':4,
            'Tamarania,Anafola':5,
            'Tamarania,Glaser':6,
            'Tamarania,Starfex':8,
            'Tamarania,Tutalu':10,
            'Tamariza,Pafiz':3,
            'Tamariza,Scetcher':6,
            'Tamariza,Tutalu':6,
            'Tamariza,Demetry':8,
            'Tamariza,Anafola':10,
            'Tamariza,Benfriege':10,
            'Scetcher,Demetry':3,
            'Scetcher,Fodker':4,
            'Scetcher,Pafiz':5,
            'Tutalu,Pafiz':7,
            'Tutalu,Benfriege':12,
            'Zinzibar,Enfeir':1,
            'Zinzibar,Starfex':10,
            'Kubani,Demetry':10,
            'Anafola,Glaser':12}

Base = dc(origBase)

def determineSkirmish(add1=True, verbose=True):
    if add1: Round[0] += 1
    skirmCities = set()
    for skirm, count in Base.items():
        if type(count) is int:
            Base[skirm] -= 1
            if Base[skirm] == 0:
                Base[skirm] = 'skirmish 1'
                skirmCities = skirmCities.union(set(skirm.split(',')))
        elif count == 'skirmish 1':
            Base[skirm] = 'skirmish 2'
            skirmCities = skirmCities.union(set(skirm.split(',')))
        else:
            Base[skirm] = origBase[skirm]
    if verbose:
        print("Cities in Skirmish on Round "+str(Round[0])+": ")
        for i in sorted(list(skirmCities)):
            print(i)
            
for i in range(Round[0]):
    determineSkirmish(False, False)

def randofsum_unbalanced(s, n):
    r = np.random.rand(n)
    a = np.array(np.round((r/np.sum(r))*s,0),dtype=int)
    while np.sum(a) > s:
        a[np.random.choice(n)] -= 1
    while np.sum(a) < s:
        a[np.random.choice(n)] += 1
    return a

def randofsum(s, n):
    return np.random.multinomial(s,np.ones(n)/n,size=1)[0]

def isint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
    
def genstats(s, atr=None, boost=0):
    if type(s) is list:
        s = np.random.choice(np.arange(s[0], s[1]+1))
    if atr is None:
        lbls = default
    elif type(atr) is int:
        lbls = default[:atr] + default[(atr+1):]
    else:
        lbls = atr
    conditions = [None, None, None]
    n = len(lbls)
    a = randofsum(s, n)
    if ('Hit Points' in lbls):
        hi = lbls.index('Hit Points')
        conditions[0] = 'a[hi]==0'
    if ('Attack' in lbls) and ('Stability' in lbls):
        ai, si = lbls.index('Attack'), lbls.index('Stability')
        conditions[1] = 'a[si]>a[ai]'
    if ('Attack' in lbls):
        ai = lbls.index('Attack')
        conditions[2] = 'a[ai]==0'
    while True:
        a = randofsum(s, n)
        breakout = True
        for i in range(len(conditions)):
            if (conditions[i] is not None) and eval(conditions[i]):
                breakout = False
                break
        if breakout: break
    print("---- Combat Level: "+str(sum(a))+" ----")
    for i in range(len(a)): print(lbls[i]+': '+str(a[i]))
    if boost > 0:
        print("NPC has additional boosted skills in: ")
        b = randofsum(boost, n)
        for i in range(len(b)):
            if b[i] > 0:
                print(lbls[i]+': +'+str(b[i]))
    else:
        b = np.zeros((n,),dtype=int)
    return a + b

def confirmStats():
    response = ''
    while response != 'y':
        print("Are these still your current stats?")
        for i in range(len(default)):
            print(default[i]+' ('+acronym[i]+'): '+str(combat[i]))
        response = ''
        while response not in {'y','n'}:
            response = input("y/n: ")
        if response == 'n':
            valid = False
            while valid==False:
                valid = True
                change = input("Input the acronym and its value that have changed separated by commas (e.g. type s=0 if stability is now 0): ")
                for val in change.split(','):
                    acr, lvl = val.split('=')
                    if (acr not in acronym) or (not isint(lvl)):
                        valid = False
                        print("Uh oh, something was typed incorrectly, try again.")
                        break
            # Once we know everything is inputted correctly, then make the changes.
            for val in change.split(','):
                acr, lvl = val.split('=')
                combat[acronym.index(acr)] = int(lvl)

def fight(npcmb=None, boost=0):
    if npcmb is None:
        npcmbstr = input("NPC's Combat level (input range with comma): ")
        if ',' in npcmbstr:
            rl = npcmbstr.split(',')
            npcmb = np.random.choice(np.arange(int(rl[0]), int(rl[1])+1))
        else:
            npcmb = int(npcmbstr)
    attackstyle = False
    while attackstyle not in {'p', 'w', 'e', 't'}:
        attackstyle = input("What is NPC's attack style? (p=physical, w=wizard, e=elemental, t=trooper): ")
    npstats = genstats(npcmb, boost=boost)
    cont = input("Would you like to fight? (y)/n/fix:")
    if cont == 'fix':
        print("Options to fix: ")
        for i in range(len(default)): print(default[i]+' ('+acronym[i]+')')
        valid = False
        while valid == False:
            valid = True
            change = input("Input the acronym and its value that should be fixed separated by commas (e.g. type c=12 if cunnning should be 12): ")
            for val in change.split(','):
                acr, lvl = val.split('=')
                if (acr not in acronym) or (not isint(lvl)):
                    valid = False
                    print("Uh oh, something was typed incorrectly, try again.")
                    break
        for val in change.split(','):
            acr, lvl = val.split('=')
            i = acronym.index(acr)
            npstats[i] = int(lvl)
            print(default[i]+" has been changed to "+str(lvl))
    elif cont == 'n': 
        return
    
    eng = input("Is NPC engager? (y)/n:")
    
    agility = npstats[0], combat[0]
    startagil = npstats[0] if eng == 'n' else npstats[0] + 2
    if startagil > combat[0]:
        order = [0]*max([1, int(npstats[0]/max([1,combat[0]]))])+[1]*max([1, int(agility[1]/max([1,agility[0]]))])
    elif combat[0] > startagil:
        order = [1]*max([1,int(combat[0]/max([1,npstats[0]]))])+[0]
    else:
        f = np.random.choice(np.arange(2))
        order = [f]*max([1,int(agility[f]/max([1,agility[1-f]]))]) + [1-f]*max([1,int(agility[1-f]/max([1,agility[f]]))])
    fighters = np.array(["NPC", "Player"])
    print("Order of Attack: "+str(list(fighters[order])))
    
    agility = npstats[0], combat[0]
    cunning = npstats[1], combat[1]
    technique = npstats[2], combat[2]
    hitpoints = [npstats[3], combat[3]]
    attack = npstats[4], combat[4]
    stability = npstats[5], combat[5]
    defense = npstats[defpos[combat[-1]]], combat[defpos[attackstyle]]
    
    cont=''
    while cont != 'n':
        for attacker in order:
            print()
            print('Attacker is '+fighters[attacker])
            # Conduct Attack
            startatk = 0 if technique[attacker]==0 else 1
            chances = 1 if technique[attacker] < 2 else technique[attacker]
            atk = np.max(np.random.choice(np.arange(startatk, attack[attacker]+1),chances))
            if (atk > stability[attacker]) and (np.random.choice(2) == 0): atk = stability[attacker]
            dcun = np.random.choice(np.arange(0, cunning[1-attacker]+1))
            print(fighters[attacker]+' attacks:  '+str(atk)+', '+fighters[1-attacker]+' outwits: '+str(dcun))
            A = np.max([0, atk-dcun])
            
            agl = int(np.random.choice(np.arange(1,agility[1-attacker]+1))/2) if agility[1-attacker]>0 else 0
            startdef = 0 if stability[1-attacker]==0 else 1
            df = np.random.choice(np.arange(startdef,defense[1-attacker]+1)) if defense[1-attacker]>0 else 0
            acun = np.random.choice(np.arange(0, cunning[attacker]+1))
            print(fighters[1-attacker]+' dodges '+str(agl)+', defends: '+str(df)+', '+fighters[attacker]+' outwits: '+str(acun))
            D = np.max([0, agl+df-acun])
            damage = np.min([hitpoints[1-attacker], np.max([0, A-D])])
            print("---------------------------------")
            hitpoints[1-attacker] -= damage
            
            if hitpoints[1-attacker]==0:
                print(fighters[1-attacker]+' faints.')
                cont ='n'
                break
            cont = input(fighters[1-attacker]+' takes '+str(damage)+' damage. Continue? (y)/n: ')
            if cont == 'n':
                break
    if hitpoints[0] == 0:
        return 'win'
    else:
        return 'lose'
            
def conductbattle(cmbt=None, boost=0):
    confirmStats()
    result = fight(cmbt, boost)
    if result == 'win':
        print("Great, you won!")
        critthink = inputskill('critical thinking')
        if rbtwn(1,12) <= critthink:
            print("Your critical thinking is successful, gain 1xp if lvl<6. Choose which stats to increase as long as opponent is not very weak.")
        else:
            print("You are unable to choose your own stats. Choose randomly betwen 1-6: 1=agility, 2=cunning, 3=hit points, 4=stability, 5=attack, 6=defense (of attacker's style)")
            
attributes = np.array(['agility','hit points','stability','cunning','attack','technique','def-physical','def-elemental','def-wizard','def-trooper'])
books = np.array(['critical Thinking',
                  'bartering',
                  'persuasion',
                  'crafting',
                  'heating',
                  'smithing',
                  'stealth',
                  'survival',
                  'gathering',
                  'excavating'])
ores = np.array(['lead','tin','copper','iron','tantalum','aluminum','kevlium','nickel','tungsten','titanium','diamond','chromium','shinopsis','ebony','astatine','promethium'])

def rbtwn(mn, mx, amt=1):
    size = None if amt <= 1 else amt
    return np.random.choice(np.arange(mn, mx+1), size) 

def chooseResult(mn, mx, amt):
    if amt<=1:
        return rbtwn(mn, mx, amt)
    choices = set(list(np.random.choice(np.arange(mn, mx+1), amt)))
    print("You have the following choices: "+str(sorted(list(choices))))
    choice = ''
    while choice not in choices:
        choice = inputint("Which of the results do you choose? ")
    return choice
    
def inputint(message):
    out = ''
    while not isint(out):
        out = input(message)
    return int(out)

def inputskill(skill):
    fat = inputint("What is your fatigue? ") if fatigue[1] is None else fatigue[0]
    fatigue[0], fatigue[1] = fat, 0
    if (rbtwn(1, 10) <= fat):
        print("Fatigue prevents you from completely activating your skill.")
    else:
        print("Successfully activate "+skill+' skill (gain 1 xp if lvl <3).')
    return np.max([0, inputint("What level is your "+skill+" skill? ") - fat])

def isconsequence(tile):
    res = ''
    while (res != 'y') and (res != 'n'):
        res = input("Did you just enter the "+tile+" this action or remained there for one action? y/n: ")
    return res == 'y'

def excavate(tile):
    ex = input("Excavate "+tile+"? (y)/n: ")
    return ex != 'n'

def excSuccess(item):
    print("Success! You found "+item+". Gain +1xp for excavating if lvl<6.")

def isbetween(mn, mx, numb):
    return (numb >= mn) * (numb <= mx)

def train(trainer):
    start = ''
    while start not in {'y','n'}:
        start = input("Begin training with "+trainer+" trainer? y/n: ")
    if start == 'n':
        return
    lvl = inputint("What is your base level? ")
    if trainer == 'adept':
        if lvl >= 8:
            print("Trainer cannot train you.")
            return
        print("Once training is successful, it will cost you "+str(4+lvl+1)+" coins.")
        cont = 'y'
        while cont != 'n':
            fatigue = inputint("What is your fatigue? ")
            if rbtwn(1,10) <= fatigue:
                print("You were unable to keep up with training.")
            else:
                if rbtwn(1,12) <= 5:
                    print("You successfully learn the skill.")
                    return
                else:
                    print("You were initially unsuccessful but your critical thinking could save you.")
                    critthink = inputskill("critical thinking")
                    if rbtwn(1,12) <= critthink:
                        print("You successfully learn the skill, gain +1xp for critical thinking if lvl<6")
                        return
                    else:
                        print("You were unsuccessful in learning the skill.")
            cont = input("Continue on next action? (y)/n: ")
        if cont == 'n':
            print("For the time the trainer gave to you, pay them "+str(int((4+lvl+1)/3))+" coins.")
    elif trainer == 'master':
        print("Once training is successful, it will cost you "+str(10+lvl+1)+" coins.")
        cont = 'y'
        while cont != 'n':
            fatigue = inputint("What is your fatigue? ")
            if rbtwn(3,10) <= fatigue:
                print("You were unable to keep up with training.")
            else:
                if lvl < 8:
                    print("You successfully learn the skill.")
                    return
                else:
                    if rbtwn(1,4) == 1:
                        print("You successfully learn the skill.")
                        return
                    else:
                        print("You initially fail to learn the skill, but critical thinking could save you.")
                        critthink = inputskill("critical thinking")
                        if rbtwn(1,16) <= critthink:
                            print("You successfully learn the skill, gain +2xp for critical thinking if lvl<8")
                            return
                        else:
                            print("You were unsuccessful in learning the skill.")
            cont = input("Continue on next action? (y)/n: ")
        if cont == 'n':
            print("For the time the trainer gave to you, pay them "+str(int((10+lvl+1)/3))+" coins.")
    elif trainer == 'outskirt':
        print("Once the training is successful, it will cost you "+('8' if lvl<8 else '12')+" coins.")
        cont = 'y'
        while cont != 'n':
            fatigue = inputint("What is your fatigue? ")
            if rbtwn(5,10) <= fatigue:
                print("You were unable to keep up with training.")
            else:
                if lvl < 8:
                    print("You successfully learn the skill.")
                    return
                else:
                    if rbtwn(1,4) == 1:
                        print("You successfully learn the skill.")
                        return
                    else:
                        print("You initially fail to learn the skill, but critical thinking could save you.")
                        critthink = inputskill("critical thinking")
                        if rbtwn(1,16) <= critthink:
                            print("You successfully learn the skill, gain +2xp for critical thinking if lvl<8")
                            return
                        else:
                            print("You were unsuccessful in learning the skill.")
            cont = input("Continue on next action? (y)/n: ")
        if cont == 'n':
            print("If outskirt trainer is a monk then they don't ask for any money. Otherwise pay 2 coins for their time.")
                
def roads(number):
    # First: Highway Robber
    if rbtwn(1, 6) <= number:
        fightrobber = True
        activatest = input("Robber approaches! Activate stealth? (y)/n: ")
        if activatest != 'n':
            stealth = inputskill("stealth")
            if rbtwn(1,12) <= stealth:
                print("Robber successfully dodged. Gain 1xp if stealth < 6.\n")
                fightrobber = False
            else:
                print("Sneak Fails!\n")
        if fightrobber:
            attackstyles = ['Warrior', 'Trooper']
            npclvl = rbtwn(3,30)
            print("Level "+str(npclvl)+" robber attacks you with attack style of "+attackstyles[rbtwn(0,1)])
            print("If there are any traders on the tile, they run away. You get 3 coins if you win.")
            conductbattle(npclvl)
            return
    else:
        print("No highway robber appears.")
    # Second: Trader
    trader = input("Is this a traders spot? (y)/n: ")
    if (trader != 'n') and (rbtwn(1, 8) == 1):
        clrs = np.array(['green', 'purple', 'yellow'])
        print("Trader appears selling 4 items! Categories are: "+str(clrs[rbtwn(0,2,4)]))
    else:
    	print("No trader appears.")
    
def ruins():
    def springtrap():
        survival = inputskill('survival')
        if rbtwn(0, 12) > survival:
            print("You spring a trap in the ruins, take 2 hit points of damage.")
        else:
            print("You escaped a trap.")
    def ancientwizard():
        persuade = input("Ancient Wizard approaches! Persuade them to be your friend? (y)/n: ")
        if persuade != 'n':
            persuasion = inputskill('persuasion')
            if rbtwn(1,12) > persuasion:
                print("Persuasion failed!\n")
            else:
                print("Successfully persuaded wizard to be a friend! Gain 1xp if persuasion < 6.\n")
                print("You can choose to have them train you in hitpoints or heating (Master trainer) for 8 coins if lvl<8 and 12 coins if lvl<12.")
                train('outskirt')
                return
        npclvl = rbtwn(60,85)
        print("Level "+str(npclvl)+" engages you in a fight. Their hitpoints are fixed at 12.")
        print("You receive 2 old cloth if you win (of whichever city you are next to)\n")
        conductbattle(npclvl)
    if isconsequence('Ruins'):
        springtrap()
        ancientwizard()
    else:
        ex = input("Excavate ruins? (y)/n: ")
        if ex != 'n':
            excavating = inputskill('excavating')
            print("Excavation options are:")
            print("1-5: Spring trap")
            print("6-7: Find old cloth")
            print("8-10: Ancient wizard approaches (lvl 60-85)")
            print("11-14: Find old tattered book")
            print("otherwise: Tiles is emptied.")
            r = chooseResult(1, 19, excavating+1)
            if isbetween(1,5,r):
                springtrap()
            elif isbetween(6,7,r):
                print("Gain 1xp if excavating < 6. You find an old cloth (of neighboring city).")
            elif isbetween(8,10,r):
                ancientwizard()
            elif isbetween(11,14,r):
                book = books[rbtwn(0,9)]
                print("Success! Increase excavating by xp+1 if lvl<6")
                print("An old tattered "+book+" book is found, you can possibly pick out some words.")
                ct = inputskill('critical thinking')
                if ct >= rbtwn(0,12):
                    print("Critical thinking successful! Gain 1xp if <6, and gain 2 levels for "+book+" if <8.")
                else:
                    print("Critical thinking unsuccessful, but you still gain 2xp of "+book+" if less than 6.")
            else:
                print("You find nothing, tile is emptied for 6 rounds!")
                
def ponds():
    ex = input("Excavate pond? (y)/n: ")
    if ex != 'n':
        excavating = inputskill('excavating')
        print("Excavation options: ")
        print("1-5: Go fishing")
        print("6-8: Find clay")
        print("9: Giant serpent attacks (lvl 20-45)")
        print("otherwise the tile is emptied")
        r = chooseResult(1,12,excavating+1)
        if isbetween(1,5,r):
            print("You find a nice fishing spot. Increase excavation xp+1 if lvl<6")
            gathering = inputskill("gathering")
            fish = 1
            while rbtwn(0, 15) <= gathering:
                fish += 1
            print("You gathered "+str(fish)+" raw fish.")
        elif isbetween(6,8,r):
            print("You found clay. Increase excavation xp+1 if lvl<6")
        elif r==9:
            npclvl = rbtwn(20,45)
            print("You encounter a giant serprent level "+str(npclvl)+" using elemental attack.\n")
            conductbattle(npclvl)
        else:
            print("You find nothing. Tile is emptied for 6 turns.")

def plains():
    if isconsequence("Plains"):
        survival = inputskill("survival")
        if rbtwn(0,12) > survival:
            print("A trap is triggered which claws at you. You take 1 fatigue damage.")
        else:
            print("You successfully dodged a trap. Gain 1 xp if less than level 6.")
    else:
        if excavate('plains'):
            excavating = inputskill('excavating')
            print("Excavate optins: ")
            print("1: Find huntsman.")
            print("2-6: Find a wild herd.")
            print("otherwise tile is emptied.")
            r = chooseResult(1,9,excavating+1)
            if r==1:
                persuasion = inputskill("persuasion")
                if rbtwn(1,12) <= persuasion:
                    print("You successfully pursuade the huntsman to teach you in agility or gathering (master trainer) for 8 coins if lvl<8 and 12 coins if lvl<12. Gain 1xp of gathering if less than 6.")
                    train('outskirt')
                else:
                    print("You failed to convince the huntsman to teach you.")
            elif isbetween(2,6,r):
                print("You find a wild animal herd. You gain 1xp for successful excavation if lvl<6")
                gathering = inputskill("gathering")
                meat = 1
                while rbtwn(0,15) <= gathering:
                    meat += 1
                print("You gather "+str(meat)+" raw meat.")
            else:
                print("You find nothing. Tile is emptied for 6 turns.")
                
def oldlibrary():
    def findHermit():
        print("A hermit is found.")
        persuasion = inputskill('persuasion')
        if rbtwn(1,12) > persuasion:
            atkstyle = ['Wizard','Elemental'][rbtwn(0,1)]
            npclvl = rbtwn(55,75)
            print("Persuasion fails. Level "+str(npclvl)+" hermit becomes angry and attacks you with "+atkstyle+" based attacks (fixed lvl 12 cunning). You get 7 coins if you win.")
            conductbattle(npclvl)
        else:
            print("Persuasion succeeds! Gain 1 xp if persuasion level < 6.")
            print("The hermit agrees to train you in either cunning or critical thinking for 8 coins if lvl<8 and 12 coins if lvl<12.")
            train('outskirt')
    if isconsequence("Old Library"):
        if rbtwn(1,4) == 1:
            findHermit()
        else:
            print("No consequence of entering old library.")
    elif excavate("old library"):
        excavating = inputskill("excavating")
        print("Excavate options: ")
        print("1: Critical Thinking book")
        print("2-3: Bartering book")
        print("4-6: Persuasion book")
        print("7-9: Crafting book")
        print("11-13: Heating book")
        print("14-16: Smithing book")
        print("17: Stealth book")
        print("18-20: Survival book")
        print("21-23: Gathering book")
        print("24-25: Excavating book")
        print("26-35: Hermit is found")
        print("otherwise: the tile is emptied")
        r = chooseResult(1, 47, excavating+1)
        if r==1:
            print("You find book on critical thinking, +2xp on excavating if lvl<6")
        elif isbetween(2,3,r):
            print("You find book on bartering, +1xp on excavating if lvl<6")
        elif isbetween(4,6,r):
            print("You find a book on persuasion, +1xp on excavating if lvl<6")
        elif isbetween(7,9,r):
            print("You find a book on crafting, +1xp on excavating if lvl<6")
        elif isbetween(11,13,r):
            print("You find a book on heating, +1xp on excavating if lvl<6")
        elif isbetween(14,16,r):
            print("You find a book on smithing, +1xp on excavating if lvl<6")
        elif r==17:
            print("You find a book on stealth, +1xp on excavating if lvl<6")
        elif isbetween(18,20,r):
            print("You find a book on gathering, +1xp on excavating if lvl<6")
        elif isbetween(21,23,r):
            print("You find a book on excavating, +1xp on excavating if lvl<6")
        elif isbetween(26,35,r):
            findHermit()
        else:
            print("You find nothing. The tile is emptied for 6 round.")
            
def cave(tier):
    def fall(tier):
        survival = inputskill("survival")
        dmg = ['0', '1', '3', '5'][tier]
        mx = [0, 4, 8, 12][tier]
        if rbtwn(1, mx) > survival:
            print("You take "+dmg+" amount of damage for slipping in the dark.")
        else:
            print("You successfully survive traversing the caves. Gain 1xp for survival if lvl<6")
    def encounter(tier):
        print("Monster approaches ... ")
        stealth = inputskill("stealth")
        mx = [0, 4, 8, 12][tier]
        lvl = [[3,20], [15, 40], [35, 60]][tier]
        rwd = ['1 hide and 1 raw meat', '3 hide and 1 raw meat', '5 hide and 2 raw meat'][tier]
        if rbtwn(1, mx) > stealth:
            atk = ['Physical', 'Wizard', 'Elemental', 'Trooper'][rbtwn(0,3)]
            npclvl = rbtwn(lvl[0],lvl[1])
            print("Monster lvl "+str(npclvl)+" engages with you (stealth unsuccessful). Monster uses "+atk)
            print("You get "+rwd+" if you win.\n")
            conductbattle(npclvl)
        else:
            print("You successfully avoid the monster, increase stealth xp by 1 if lvl<6")
    tile = "caves tier "+str(tier)
    if isconsequence(tile):
        fall(tier)
        encounter(tier)
    elif excavate(tile):
        excavating = inputskill("excavating")
        print("Excavate options:")
        if tier == 1:
            print("1-6: Lead")
            print("7-12: Tin")
            print("13-20: Monster encounter lvl 3-20")
            print("otherwise: tile is emptied.")
            r = chooseResult(1, 28, excavating+1)
            if isbetween(1,6, r):
                excSuccess("lead")
            elif isbetween(7,12,r):
                excSuccess("tin")
            elif isbetween(13,20,r):
                encounter(tier)
            else:
                print("Nothing found, tile is emptied for 6 rounds")
        elif tier == 2:
            print("1-5: Tantalum")
            print("6-10: Aluminum")
            print("11-19: Monster encounter lvl 15-40")
            print("otherwise: tile is emptied.")
            r = chooseResult(1, 28, excavating+1)
            if isbetween(1,5,r):
                excSuccess("tantalum")
            elif isbetween(6,10,r):
                excSuccess("aluminum")
            elif isbetween(11,19,r):
                encounter(tier)
            else:
                print("Nothing found, tile is emptied for 6 rounds")
        elif tier == 3:
            print("1-3: Tungsten")
            print("4-6: Titanium")
            print("7-16: Monster encounter lvl 35-60")
            print("otherwise: tile is emptied.")
            r = chooseResult(1, 28, excavating+1)
            if isbetween(1,3,r):
                excSuccess("tungsten")
            elif isbetween(4,6,r):
                excSuccess("titanium")
            elif isbetween(7,16,r):
                encounter(tier)
            else:
                print("Nothing found, tile is emptied for 6 rounds")
    else:
        if tier < 3:
            descend = ''
            while (descend != 'y') and (descend != 'n'):
                descend = input("Descend down a tier? y/n: ")
            if descend=='y':
                print("Descending down to tier "+str(tier+1))
                cave(tier+1)
                return
        elif tier > 1:
            ascend = ''
            while (ascend != 'y') and (ascend != 'n'):
                ascend = input("Ascend up a tier? y/n: ")
            if ascend == 'y':
                print("Ascending up to tier "+str(tier-1))
                cave(tier-1)
                return
            
def outpost():
    def findBandit():
        print("Bandit approaches ... ")
        stealth = inputskill("stealth")
        if rbtwn(1,12) > stealth:
            print("You fail to avoid the bandit.")
            atk = ["elemental", "physical"][rbtwn(0,1)]
            npclvl = rbtwn(15,40)
            print("Bandit lvl "+str(npclvl)+" battles you with "+atk+" style attack. You get 4 coins if you win.\n")
            conductbattle(npclvl)
        else:
            print("Success! You avoid the bandit, gain +1xp stealth if lvl<6.")
    if isconsequence("outpost"):
        if rbtwn(1,3)==1:
            findBandit()
        else:
            print("No consequences.")
    elif excavate("outpost"):
        excavating = inputskill("excavating")
        print("Excavate options:")
        print("1-4: string")
        print("5-7: beads")
        print("8-9: sand")
        print("10-13: Bandit approaches lvl 15-40.")
        print("otherwise the tile is emptied.")
        r = chooseResult(1, 16, excavating+1)
        if isbetween(1,4,r):
            excSuccess("string")
        elif isbetween(5,7,r):
            excSuccess("beads")
        elif isbetween(8,9,r):
            excSuccess("sand")
        elif isbetween(10,13,r):
            findBandit()
        else:
            print("You find nothing. The tile is emptied for 6 rounds.")

def mountain(tier):
    def findmonk():
        print("You find a monk. Give excavating skill +1xp if lvl<6")
        print("The monk will train you in survival or excavating for 8 coins if lvl<8 or 12 coins if lvl<12")
        train('outskirt')
    tile = 'mountain tier '+str(tier)
    if isconsequence(tile):
        survival = inputskill('survival')
        if rbtwn(1,12) > survival:
            print("The trecherous conditions of the mountain damages you "+str(tier)+" hit points and fatigue")
        else:
            print("You successfully survive the tough mountain conditions this turn. Gain 1xp for survival if lvl<6")
    elif excavate(tile):
        excavating = inputskill("excavating")
        print("Excavate options:")
        if tier == 1:
            print("1: Find a monk.")
            print("2-6: Find copper.")
            print("7-11: Find iron.")
            print("otherwise you find nothing")
            r = chooseResult(1, 21, excavating+1)
            if r==1:
                findmonk()
            elif isbetween(2,6,r):
                excSuccess("copper")
            elif isbetween(7,11,r):
                excSuccess("iron")
            else:
                print("You fail to find anything. Tile is emptied for 6 rounds.")
        elif tier == 2:
            print("1-3: Find a monk.")
            print("4-7: Kevlum")
            print("8-11: Nickel")
            print("otherwise: nothing")
            r = chooseResult(1,21,excavating+1)
            if isbetween(1,3,r):
                findmonk()
            elif isbetween(4,7,r):
                excSuccess("kevlium")
            elif isbetween(8,11,r):
                excSuccess("nickel")
            else:
                print("You fail to find anything. Tile is emptied for 6 rounds.")
        elif tier == 3:
            print("1-2: Find a monk.")
            print("3-5: Diamond")
            print("6-8: Chromium")
            print("otherwise: nothing")
            r = chooseResult(1, 21, excavating+1)
            if isbetween(1,2,r):
                findmonk()
            elif isbetween(3,5,r):
                excSuccess("Diamond")
            elif isbetween(6,8,r):
                excSuccess("Chromium")
            else:
                print("You find nothing. The tile is emptied for 6 rounds.")
    else:
        if tier > 1:
            descend = ''
            while (descend != 'y') and (descend != 'n'):
                descend = input("Descend the mountain? y/n: ")
            if descend == 'y':
                print("Descending mountain...")
                mountain(tier-1)
                return
        elif tier < 3:
            ascend = ''
            while (ascend != 'y') and (ascend != 'n'):
                ascend = input("Ascend the mountain? y/n: ")
            if ascend == 'y':
                print("Ascending mountain...")
                mountain(tier+1)
                return
            
def wilderness():
    if isconsequence('wilderness'):
        survival = inputskill("survival")
        if rbtwn(1,16) > survival:
            print("Player takes 3 hit points and 2 fatigue of damage by poisonous vines")
        else:
            print("Successfully traverse through the wilderness unharmed. Gain +1xp survival if lvl<8")
            stealth = inputskill("stealth")
            if rbtwn(1,14) > stealth:
                npclvl = rbtwn(95,120)
                print("Wild vine monster lvl "+str(npclvl)+" attacks you with elemental attack. You get 4 bark if you win.")
                conductbattle(npclvl)
            else:
                print("Successfully sneak past the vine monster. Gain +1xp for stealth if lvl<8")
    elif excavate('wilderness'):
        excavating = inputskill("excavating")
        print("Excavate options:")
        print("1: Shinopsis")
        print("2: Ebony")
        print("3: Astatine")
        print("4: Promethium")
        print("5: Gem")
        print("6-16: You find a grand tree")
        print("otherwise: nothing")
        r = chooseResult(1,33,excavating+1)
        if r==1:
            excSuccess("Shinopsis")
        elif r==2:
            excSuccess("Ebony")
        elif r==3:
            excSuccess("Astatine")
        elif r==4:
            excSuccess("Promethium")
        elif r==5:
            excSuccess("Gem")
        elif isbetween(6,16,r):
            gathering = str(inputskill("gathering"))
            if rbtwn(0,1)==0:
                print("Gather "+gathering+" fruits.")
            else:
                print("Gather "+gathering+" bark.")
        else:
            print("You find nothing.")
        print("Wilderness emptied for 6 rounds.")
        
def battlezone():
    def ninjas():
        stealth = inputskill("stealth")
        if rbtwn(1,16) > stealth:
            print("You were not able to avoid the ninja warriors.")
            r = rbtwn(1,4)
            if r == 1:
                cmbt, st = str(rbtwn(65,90)), str(rbtwn(6,8))
                print("You encounter a lvl "+cmbt+" ninja with stealth "+st+". You get 10 coins if you win.")
                conductbattle(int(cmbt))
            elif r == 2:
                cmbt, st = str(rbtwn(45,70)), str(rbtwn(5,6))
                print("You encounter a lvl "+cmbt+" ninja with stealth "+st+". You get 7 coins if you win.")
                conductbattle(int(cmbt))
            elif r == 3:
                cmbt1, cmbt2 = str(rbtwn(35,60)), str(rbtwn(35,60))
                print("You encounter a lvl "+cmbt1+" and a lvl "+cmbt2+" ninja of stealth 5. You get 8 coins if you win.")
                conductbattle(int(cmbt1)+int(cmbt2))
            elif r == 4:
                cmbt1, cmbt2, cmbt3 = str(rbtwn(25,50)),str(rbtwn(25,50)),str(rbtwn(25,50))
                print("You encounter lvl "+cmbt1+", "+cmbt2+", and "+cmbt3+" ninjas with 3 stealth. You get 9 coins if you win.")
                conductbattle(int(cmbt1)+int(cmbt2)+int(cmbt3))
        else:
            print("You successfully avoided conflict with the ninja. Increase stealth +2xp if lvl<8.")
    skirmish = ''
    while (skirmish != 'y') and (skirmish != 'n'):
        skirmish = input("Are Zinzibar and Enfeir currently in a skirmish? y/n: ")
    if skirmish == 'n':
        print("You cannot leave nor enter the battle zone while skirmish is inactive.")
    elif isconsequence('battle zone'):
        ninjas()
    elif excavate("battle zone"):
        excavating = inputskill("excavating")
        print("Excavte options:")
        print("1-16: Find ore (respective to their smithing id)")
        print("17-29: Engage with ninjas.")
        print("otherwise: you find nothing")
        r = chooseResult(1, 40, excavating+1)
        if isbetween(1,16,r):
            excSuccess(ores[r])
        elif isbetween(17,29,r):
            ninjas()
        else:
            print("You find nothing. tile is NOT emptied.")
            
def city():
    print("Action options: ")
    print("Smithing quarters (s)")
    print("Read knowledge book (r)")
    print("Train with adept trainers (a)")
    print("Train with master trainers (m)")
    print("Barter in the market (b)")
    print("Look for a worker to hire to automate your market (w)")
    print("(Only in Scetcher) Enter duel arena (d)")
    print("(Only in Scetcher) Enter tournament (t)")
    allowed = {'s','a','m','w','d','t','b'}
    action = ''
    while action not in allowed:
        action = input("What would you like to do? ")
    if action == 'w':
        mrkt = ''
        while mrkt not in {'y', 'n'}:
            mrkt = input("Do you own a market stand in this city? y/n: ")
        if mrkt == 'n':
            print("You can't look for a worker until you own a market stand.")
            return
        excavating = inputskill("excavating")
        if rbtwn(1,6) <= excavating:
            print("You find a potential worker. Gain 1xp excavating if lvl<6. Now you need to convince him to work for you.")
            persuasion = inputskill("persuasion")
            if rbtwn(1,6) <= persuasion:
                print("You successfully convince the worker to automate your shop! Whenever your not in the city he will work for you (your income is subtracted by 1).")
            else:
                print("You can't convince the man to work for you.")
        else:
            print("You fail to find anyone.")
    elif action == 's':
        print("You can rent the smithing's quarters for 2 coins (unless otherwise stated) per action.")
        print("Otherwise if city has adept smithing trainer you can have him smith up to lvl8 at cost of 5 coins + cost sell cost of smithing material consumed. You must provide the smithing material")
        print("Otherwise if city has master smithing trainer you can have him smith up to lvl12 at cost of 12 coins + cost sell cost of smithing material consumed. You must provide the smithing material")
        barter = inputskill("bartering")
        if rbtwn(1,12) <= barter:
            print("You successfully barter with the smith to rent his facilities (or have him smith for you). Subtract 1 coin. Add 1xp to bartering if lvl<6.")
        else:
            print("You are not successful in bartering.")
    elif action == 'r':
        critthink = inputskill("critical thinking")
        if rbtwn(0,8) <= critthink:
            print("You are successful in learning the skill! The book is destroyed. Add 1xp to critical thinking if lvl<6.")
        else:
            print("You are unsuccessful in learning the skill, take an extra fatigue for exerting your brain. The book is not destroyed.")
    elif action == 'a':
        train('adept')
    elif action == 'm':
        train('master')
    elif action == 'b':
        barter = inputskill("bartering")
        if rbtwn(1,12) <= barter:
            print("You are successful in bartering. Prices for the next 3 items are sold for +1 coin and bought for -1 coin. Add 1xp to bartering if lvl<6.")
        else:
            print("You are unsuccessful in bartering.")
    elif action == 'd':
        cmbt = inputint("What is your combat level? ")
        style = ['physical','wizard','elemental','trooper'][rbtwn(0,3)]
        npclvl = rbtwn(cmbt, cmbt+6)
        boost = rbtwn(0, int(npclvl/5))
        print("You face a foe lvl "+str(npclvl)+" using "+style+' style of attack with a boost of '+str(boost)+'. You get '+str(int(cmbt/10)+2)+' coins for winning.')
        conductbattle(npclvl, boost=boost)
    elif action == 't':
        print("Not yet implemented.")
        
def computeAction(recursion, direct=False):
    options = {'city':city,'b':battlezone,'w':wilderness,'u':ruins,'o':outpost,
               'r1':lambda: roads(1),'r2':lambda: roads(2),'r3':lambda: roads(3),'r4':lambda: roads(4),'r5':lambda: roads(5),'r6':lambda: roads(6),
               'm1':lambda: mountain(1),'m2':lambda: mountain(2),'m3':lambda: mountain(3),
               'c1':lambda: cave(1),'c2':lambda: cave(2),'c3':lambda: cave(3), 'fight':conductbattle,
               'l':oldlibrary,'p':plains,'f':ponds,'up':determineSkirmish}
    if not direct:
        input("Click enter to continue to main menu")
        print()
    print("Tile options: ")
    print("City (city)")
    print("Road (r+number)")
    print("Fishing Pond (f)")
    print("Plains (p)")
    print("Caves (c+tier)")
    print("Mountains (m+tier)")
    print("Outposts (o)")
    print("Old Lirbraries (l)")
    print("Ruins (u)")
    print("Battle Zones (b)")
    print("Wilderness (w)")
    print("Update Round (up)")
    print("Fight (fight)")
    print("Quit (q)")
    tile = ''
    while tile not in options:
        tile = input("Which tile are you on? ")
        if tile == 'q':
            return
    options[tile]()
    fatigue[1] = None
    if recursion:
        computeAction(recursion)

frameinfo = getframeinfo(currentframe())
#computeAction(True, True)