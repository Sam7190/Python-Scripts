"""
cd Documents\\GitHub\\Python-Scripts\\elementalSword
python game.py

"""

import numpy as np
import os
import sys
import socket_client
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
# to use buttons:
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Ellipse
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.core.window import Window

# Tile Resolution and number of tiles in gameboard.
seed = 303
xpix = 343
ypix = 396
xtiles = 17
ytiles = 15
cities = {'anafola':{'Combat Style':'Summoner','Coins':3,'Knowledges':[('Excavating',1),('Persuasion',1)],'Combat Boosts':[('Stability',2)]},
          'benfriege':{'Combat Style':'Elemental','Coins':2,'Knowledges':[('Crafting',2)],'Combat Boosts':[('Stability',1),('Cunning',1)]},
          'demetry':{'Combat Style':'Elemental','Coins':9,'Knowledges':[('Bartering',2)],'Combat Boosts':[]},
          'enfeir':{'Combat Style':'Physical','Coins':3,'Knowledges':[('Stealth',2)],'Combat Boosts':[('Cunning',2)]},
          'fodker':{'Combat Style':'Trooper','Coins':5,'Knowledges':[('Stability',2)],'Combat Boosts':[('Def-Physical',3)]},
          'glaser':{'Combat Style':'Elemental','Coins':2,'Knowledges':[('Survival',1)],'Combat Boosts':[('Def-Trooper',3),('Cunning',1)]},
          'kubani':{'Combat Style':'Physical','Coins':3,'Knowledges':[('Crafting',1),('Gathering',1)],'Combat Boosts':[('Def-Wizard',3),('Agility',1)]},
          'pafiz':{'Combat Style':'Wizard','Coins':4,'Knowledges':[('Persuasion',1)],'Combat Boosts':[('Def-Elemental',3)]},
          'scetcher':{'Combat Style':'Physical','Coins':4,'Knowledges':[],'Combat Boosts':[('Attack',1),('Hit Points',1),('Def-Physical',1),('Def-Wizard',1),('Def-Elemental',1),('Def-Trooper',1)]},
          'starfex':{'Combat Style':'Elemental','Coins':5,'Knowledges':[('Heating',1),('Gathering',1)],'Combat Boosts':[('Attack',2)]},
          'tamarania':{'Combat Style':'Physical','Coins':7,'Knowledges':[('Smithing',1)],'Combat Boosts':[('Attack',2)]},
          'tamariza':{'Combat Style':'Wizard','Coins':5,'Knowledges':[('Critical Thinking',1)],'Combat Boosts':[('Def-Physical',1),('Def-Elemental',2),('Hit Points',1)]},
          'tutalu':{'Combat Style':'Trooper','Coins':3,'Knowledges':[('Excavating',2)],'Combat Boosts':[('Attack',3)]},
          'zinzibar':{'Combat Style':'Physical','Coins':2,'Knowledges':[('Stealth',2)],'Combat Boosts':[('Agility',2),('Attack',1)]}}

def make_positions(start_y=0):
    pos = {}
    for i in range(start_y, ytiles):
        for j in range(xtiles):
            tiletype = input(f'({j},{i}): ')
            if tiletype != '':
                if tiletype not in pos:
                    pos[tiletype] = [(j, i)]
                else:
                    pos[tiletype].append((j, i))
    return pos

def merge_positions(pos1, pos2):
    pos = {}
    for key in pos1:
        if key not in pos2:
            pos[key] = pos1[key]
        else:
            seen_tup = set()
            pos[key] = []
            for tup in pos1[key]:
                pos[key].append(tup)
                seen_tup.add(tup)
            for tup in pos2[key]:
                if tup not in seen_tup:
                    pos[key].append(tup)
    for key in pos2:
        if key not in pos:
            pos[key] = pos2[key]
    return pos

positions = {'road': [(3, 1),(4, 1),(5, 1),(6, 1),(8, 1),(9, 1),(10, 1),(11, 2),(12, 2),(13, 2),(2, 2),(3, 2),(3, 3),(1, 3),(1, 4),(1, 5),(2, 5),(3, 5),
                      (4, 6),(8, 2),(7, 3),(14, 3),(5, 4),(6, 4),(7, 4),(9, 4),(10, 4),(11, 4),(12, 4),(15, 4),(6, 5),(10, 5),(12, 5),(14, 5),(6, 6),(11, 6),
                      (13, 6),(14, 6),(1, 7),(5, 7),(6, 7),(7, 7),(8, 7),(9, 7),(10, 7),(12, 7),(2, 8),(4, 8),(12, 8),(2, 9),(3, 9),(4, 9),(12, 9),(5, 10),
                      (13, 10),(5, 11),(13, 11),(6, 12),(13, 12)],
             'randoms': [(4, 2),(5, 2),(6, 2),(2, 3),(9, 2),(10, 2),(11, 1),(5, 3),(6, 3),(9, 3),(13, 3),(2, 4),(13, 4),(14, 4),(5, 5),(9, 5),(11, 5),(13, 5),
                         (3, 6),(7, 6),(8, 6),(9, 6),(10, 6),(2, 7),(13, 7),(1, 8),(3, 8),(6, 8),(7, 8),(8, 8),(9, 8),(10, 8),(13, 8),(14, 8),(15, 8),(5, 9),
                         (6, 9),(7, 9),(8, 9),(9, 9),(10, 9),(11, 9),(13, 9),(14, 9),(4, 10),(6, 10),(11, 10),(12, 10),(14, 10),(3, 11),(4, 11),(6, 11),
                         (11, 11),(12, 11),(4, 12),(7, 12),(12, 12)],
             'ruins': [(1, 1), (2, 0), (3, 0), (14, 0), (15, 1)],
             'battle1': [(14, 2)], 'battle2': [(14, 1)],
             'village1': [(0, 5),(7, 0),(4, 3),(12, 1),(8, 3),(10, 3),(15, 3),(5, 6),(12, 6),(14, 7),(1, 9),(5, 12),(12, 13)],
             'village2': [(8, 0),(0, 6),(16, 2),(12, 3),(3, 4),(7, 5),(3, 7),(15, 7),(11, 8),(1, 10),(4, 13),(13, 14)],
             'village3': [(7, 2),(4, 5),(8, 5),(16, 6),(0, 7),(5, 8),(1, 11),(5, 14),(14, 14)],
             'village4': [(15, 5), (2, 6), (2, 11), (14, 13), (6, 14)],
             'village5': [(3, 10), (14, 12), (6, 13)],
             'fodker': [(2, 1)],'kubani': [(7, 1)],'scetcher': [(4, 4)],'pafiz': [(1, 6)],'tamariza': [(4, 7)],'zinzibar': [(15, 2)],'enfeir': [(13, 1)],
             'tamarania': [(11, 3)],'demetry': [(8, 4)],'starfex': [(15, 6)],'anafola': [(11, 7)],'tutalu': [(2, 10)],'benfriege': [(5, 13)],'glaser': [(13, 13)],
             'wilderness': [(7, 10),(8, 10),(9, 10),(10, 10),(7, 11),(8, 11),(9, 11),(10, 11),(8, 12),(9, 12),(10, 12),(11, 12),(7, 13),(11, 13)]}
randoms = ['oldlibrary','outpost','cave','mountain','plains','pond']

def get_dim(dx, dy):
    return int(xpix*dx), int((ypix * (dy-(dy//2))) + (xpix * (dy//2) / np.sqrt(3)))

xsize, ysize = get_dim(xtiles, ytiles)
scale = 1
xprel, yprel = scale * (xpix / xsize), scale * (ypix / ysize)

def get_pos(x, y):
    if y % 2:
        xoffset, yoffset = xpix/2, (xpix/np.sqrt(3)) + (ypix/2 - xpix/(2*np.sqrt(3)))
    else:
        xoffset, yoffset = 0, 0
    return (x * xpix) + xoffset, (y // 2) * (ypix + xpix/np.sqrt(3)) + yoffset

def get_posint(x, y):
    px, py = get_pos(x, y)
    return int(px), int(py)

def get_relpos(x, y):
    px, py = get_pos(x, y)
    return float(px / xsize), float(py / ysize)

class Tile(ButtonBehavior, HoverBehavior, Image):
    def __init__(self, tile, x, y, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.source = f'images\\{tile}.png'
        self.parentBoard = None
        self.neighbors = set()
        self.bind(on_press=self.initiate)
        self.tile = tile
        self.gridx = x
        self.gridy = y
        xpos, ypos = get_relpos(x, y)
        self.pos_hint = {'x': xpos, 'y': ypos}
        self.size_hint = (xprel, yprel)
    def set_neighbors(self):
        neighbors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1 if self.gridy % 2 else -1, 1], [1 if self.gridy % 2 else -1, -1]]
        self.neighbors = set()
        for dx, dy in neighbors:
            x, y = self.gridx + dx, self.gridy + dy
            if (x, y) in self.parentBoard.gridtiles:
                self.neighbors.add((x, y))
    def on_enter(self, *args):
        self.source = f'selectedimages\\{self.tile}.png'
    def on_leave(self, *args):
        self.source = f'images\\{self.tile}.png'
    def initiate(self, instance):
        tiletypes = [self.parentBoard.gridtiles[(x,y)].tile for x, y in self.neighbors]
        print(self.tile, 'with neighbors: ', tiletypes)

class BoardPage(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = get_dim(xtiles, ytiles)
        self.gridtiles = {}
        for tiletype in positions:
            if tiletype == 'randoms':
                continue
            for x, y in positions[tiletype]: self.add_tile(tiletype, x, y)
        np.random.seed(seed)
        randomChoice = np.random.choice(randoms*int(np.ceil(len(positions['randoms'])/len(randoms))), len(positions['randoms']))
        for i in range(len(positions['randoms'])):
            x, y = positions['randoms'][i]
            self.add_tile(randomChoice[i], x, y)
        for T in self.gridtiles.values():
            T.set_neighbors()
    def add_tile(self, tiletype, x, y):
        T = Tile(tiletype, x, y)
        self.gridtiles[(x,y)] = T
        T.parentBoard = self
        self.add_widget(T)
#        size_hint = (xprel, yprel)
#        xpos, ypos = get_relpos(4, 4)
#        pos_hint = {'x':xpos, 'y':ypos}
#        with self.canvas:
#            Ellipse(segments=6,angle_start=0,angle_end=360,pos=get_posint(4,4),size=(xpix, ypix))
        
class BirthCityButton(Image, ButtonBehavior, HoverBehavior):
    def __init__(self, parentPage, city, **kwargs):
        super(BirthCityButton, self).__init__(**kwargs)
        self.parentPage = parentPage
        self.city = city
        self.source = f'images\\{self.city}.png'
    def on_enter(self, *args):
        self.source = f'selectedimages\\{self.city}.png'
        benefits = cities[self.city]
        ks, cb = benefits['Knowledges'], benefits['Combat Boosts']
        kstr = ', '.join([(str(ks[i][1])+' ' if ks[i][1]>1 else '')+ks[i][0] for i in range(len(ks))]) if len(ks)>0 else '-'
        cstr = ', '.join([(str(cb[i][1])+' ' if cb[i][1]>1 else '')+cb[i][0] for i in range(len(cb))]) if len(cb)>0 else '-'
        display = '[b][color=ffa500]'+self.city[0].upper()+self.city[1:]+'[/color][/b]:\nCombat Style: [color=ffa500]'+benefits['Combat Style']+'[/color]\nCoins: [color=ffa500]'+str(benefits['Coins'])
        display += '[/color]\nKnowledges: [color=ffa500]'+kstr+'[/color]\nCombat Boosts: [color=ffa500]'+cstr
        self.parentPage.displaylbl.text = '[color=000000]'+display+'[/color][/color]'
    def on_leave(self, *args):
        self.source = f'images\\{self.city}.png'
        
class BirthCityPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols=7
        cityorder = sorted(cities.keys())
        for city in cityorder:
            bcb = BirthCityButton(self, city=city)
            self.add_widget(bcb)
        self.add_widget(Label())
        self.add_widget(Label())
        self.add_widget(Label())
        self.displaylbl = Label(text='[color=000000]Hover over a city to see their benefits![/color]', markup=True,
                                halign='center',valign='middle')
        self.add_widget(self.displaylbl)
        
class LaunchPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1
        self.username = game_app.connect_page.username.text
        self.usernames, self.ready = [self.username], {self.username:0}
        self.label = Label(text=f'[color=000000]{self.username}: Not Ready[/color]', height=Window.size[1]*0.9, size_hint_y=None, markup=True)
        self.add_widget(self.label)
        self.readyButton = Button(text="Ready Up")
        self.readyButton.bind(on_press=self.update_self)
        self.add_widget(self.readyButton)
        socket_client.start_listening(self.incoming_message, show_error)
        socket_client.send("[LAUNCH]","Listening")
    def check_launch(self):
        all_ready = True
        for value in self.ready.values():
            if value == 0:
                all_ready = False
                break
        if all_ready:
            game_app.screen_manager.current = 'Birth City Chooser'
    def refresh_label(self):
        self.label.text = '[color=000000]'+'\n'.join([self.usernames[i]+': '+['Not Ready', 'Ready'][self.ready[self.usernames[i]]] for i in range(len(self.usernames))])+'[/color]'
        self.check_launch()
    def incoming_message(self, username, category, message):
        if category == '[LAUNCH]':
            if username not in self.ready:
                self.usernames.append(username)
            self.ready[username] = 1 if message == 'Ready' else 0
        elif (category == '[CONNECTION]') and (message == 'Closed'):
            self.usernames.remove(username)
            self.ready.pop(username)
        self.refresh_label()
    def update_self(self, _):
        self.ready[self.username] = 1 - self.ready[self.username]
        # send the message to the server that they are ready
        if self.ready[self.username]:
            socket_client.send("[LAUNCH]","Ready")
            self.readyButton.text = "Not Ready Anymore" 
        else:
            socket_client.send("[LAUNCH]","Not Ready")
            self.readyButton.text = "Ready Up"
        self.refresh_label()
        
            
            
        
# Simple information/error page
class InfoPage(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Just one column
        self.cols = 1

        # And one label with bigger font and centered text
        self.message = Label(halign="center", valign="middle", font_size=30)

        # By default every widget returns it's side as [100, 100], it gets finally resized,
        # but we have to listen for size change to get a new one
        # more: https://github.com/kivy/kivy/issues/1044
        self.message.bind(width=self.update_text_width)

        # Add text widget to the layout
        self.add_widget(self.message)

    # Called with a message, to update message text in widget
    def update_info(self, message):
        self.message.text = message

    # Called on label width update, so we can set text width properly - to 90% of label width
    def update_text_width(self, *_):
        self.message.text_size = (self.message.width * 0.9, None)
        
class ConnectPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cols = 2  # used for our grid
        if os.path.isfile("prev_details.txt"):
            with open("prev_details.txt","r") as f:
                d = f.read().split(",")
                prev_ip = d[0]
                prev_port = d[1]
                prev_username = d[2]
        else:
            prev_ip, prev_port, prev_username = '','',''

        self.add_widget(Label(text='IP:'))  # widget #1, top left
        self.ip = TextInput(text=prev_ip, multiline=False)  # defining self.ip...
        self.add_widget(self.ip) # widget #2, top right

        self.add_widget(Label(text='Port:'))
        self.port = TextInput(text=prev_port, multiline=False)
        self.add_widget(self.port)

        self.add_widget(Label(text='Username:'))
        self.username = TextInput(text=prev_username, multiline=False)
        self.add_widget(self.username)

        # add our button.
        self.join = Button(text="Join")
        self.join.bind(on_press=self.join_button)
        self.add_widget(Label())  # just take up the spot.
        self.add_widget(self.join)

    def join_button(self, instance):
        port = self.port.text
        ip = self.ip.text
        username = self.username.text
        with open("prev_details.txt","w") as f:
            f.write(f"{ip},{port},{username}")
        #print(f"Joining {ip}:{port} as {username}")
        # Create info string, update InfoPage with a message and show it
        info = f"Joining {ip}:{port} as {username}"
        game_app.info_page.update_info(info)
        game_app.screen_manager.current = 'Info'
        Clock.schedule_once(self.connect, 1)

    # Connects to the server
    # (second parameter is the time after which this function had been called,
    #  we don't care about it, but kivy sends it, so we have to receive it)
    def connect(self, _):

        # Get information for sockets client
        port = int(self.port.text)
        ip = self.ip.text
        username = self.username.text

        if not socket_client.connect(ip, port, username, show_error):
            return

        # Create chat page and activate it
        game_app.start_game_screen()
        game_app.screen_manager.current = 'Launcher'
        

        
class EpicApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        
        # Connect Page
        self.connect_page = ConnectPage()
        screen = Screen(name='Connect')
        screen.add_widget(self.connect_page)
        self.screen_manager.add_widget(screen)
        
        # Info page
        self.info_page = InfoPage()
        screen = Screen(name='Info')
        screen.add_widget(self.info_page)
        self.screen_manager.add_widget(screen)
        return self.screen_manager
        
    def start_game_screen(self):
        Window.clearcolor = (1, 1, 1, 1)
        # Launch Page
        self.launch_page = LaunchPage()
        screen = Screen(name = 'Launcher')
        screen.add_widget(self.launch_page)
        self.screen_manager.add_widget(screen)
        
        # City chooser
        self.chooseCity_page = BirthCityPage()
        screen = Screen(name='Birth City Chooser')
        screen.add_widget(self.chooseCity_page)
        self.screen_manager.add_widget(screen)
        
        # The board
        self.board_page = BoardPage()
        screen = Screen(name='Board')
        screen.add_widget(self.board_page)
        self.screen_manager.add_widget(screen)
        

# Error callback function, used by sockets client
# Updates info page with an error message, shows message and schedules exit in 10 seconds
# time.sleep() won't work here - will block Kivy and page with error message won't show up
def show_error(message):
    game_app.info_page.update_info(message)
    game_app.screen_manager.current = 'Info'
    Clock.schedule_once(sys.exit, 10)

if __name__ == "__main__":
    game_app = EpicApp()
    game_app.run()