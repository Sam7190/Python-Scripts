# Launch Server:
"""
cd Documents\\GitHub\\Python-Scripts\\elementalSword
python socket_server.py

"""

import socket
import select
import pickle
import datetime
from threading import Thread
import numpy as np
import pandas as pd

HEADER_LENGTH = 50
PORT = 1234

# List of connected clients - socket as a key, user header and name as data
clients = {}
client_code = {}
client_gameStatus = {}
game_launched = [False]

# Settings:
difficulty = [None]
seed = [None]
default_save = [None]
default_load = [None]
gameEnd = [None]

## getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
IP = socket.gethostbyname(hostname)
#IP = ""#'ec2-3-133-139-92.us-east-2.compute.amazonaws.com'


# Create a socket
# socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
# socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# SO_ - socket option
# SOL_ - socket option level
# Sets REUSEADDR (as a socket option) to 1 on socket
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind, so server informs operating system that it's going to use given IP and port
# For a server using 0.0.0.0 means to listen on all available interfaces, useful to connect locally to 127.0.0.1 and remotely to LAN interface IP
server_socket.bind((IP, PORT))

# This makes server listen to new connections
server_socket.listen()

# List of sockets for select.select()
sockets_list = [server_socket]



print(f'Listening for connections on {IP}:{PORT}...')

# Handles message receiving
def receive_message(client_socket):

    try:

        # Receive our "header" containing message length, it's size is defined and constant
        message_header = client_socket.recv(HEADER_LENGTH)

        # If we received no data, client gracefully closed a connection, for example using socket.close() or socket.shutdown(socket.SHUT_RDWR)
        if not len(message_header):
            return False

        # Convert header to int value
        message_length = int(message_header.decode('utf-8').strip())
        
        first_message = client_socket.recv(message_length)
        
        scnd_message_header = client_socket.recv(HEADER_LENGTH)
        scnd_message_length = int(scnd_message_header.decode('utf-8').strip())
        scnd_message = client_socket.recv(scnd_message_length)
        # Return an object of message header and message data
        return {'header': message_header, 'category': first_message, '2nd header': scnd_message_header, 'data':scnd_message}

    except:

        # If we are here, client closed connection violently, for example by pressing ctrl+c on his script
        # or just lost his connection
        # socket.close() also invokes socket.shutdown(socket.SHUT_RDWR) what sends information about closing the socket (shutdown read/write)
        # and that's also a cause when we receive an empty message
        return False
    
def sendMessage(username_from, username_to, category, message):
    username = username_from.encode('utf-8')
    username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
    category = category.encode('utf-8')
    category_header = f"{len(category):<{HEADER_LENGTH}}".encode('utf-8')
    if type(message) is str:
        message = message.encode('utf-8')
    else:
        message = pickle.dumps(message)
    message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
    client_socket = client_code[username_to]['socket']
    client_socket.send(username_header + username + category_header + category + message_header + message)
    
def decoded_message(user, message):
    username = user['data'].decode('utf-8')
    category = message['category'].decode('utf-8')
    try:
        msg = message["data"].decode("utf-8")
    except UnicodeDecodeError:
        msg = pickle.loads(message["data"])
    return username, category, msg

cities = ['anafola','benfriege','demetry','enfeir','fodker','glaser','kubani','pafiz','scetcher','starfex','tamarania','tamariza','tutalu','zinzibar']
connectivity = np.array([[ 0, 7, 3, 5, 7, 3, 5, 7, 5, 2, 2, 4, 6, 4],
                         [ 0, 0, 6,10, 7,10, 7, 5, 5, 9, 8, 3, 4,11],
                         [ 0, 0, 0, 4, 4, 6, 2, 4, 2, 4, 2, 3, 5, 6],
                         [ 0, 0, 0, 0, 7, 7, 4, 8, 6, 5, 2, 7, 9, 6],
                         [ 0, 0, 0, 0, 0,10, 3, 3, 2, 8, 6, 4, 5,10],
                         [12, 0, 0, 0, 0, 0, 8,10, 9, 5, 5, 7, 9, 6],
                         [ 0, 0,10, 0, 0, 0, 0, 5, 3, 6, 3, 4, 6, 7],
                         [ 0, 0, 0, 0, 8, 0, 0, 0, 2, 8, 6, 3, 2,10],
                         [ 0, 0, 6, 0, 6, 0,10, 7, 0, 6, 4, 2, 4, 8],
                         [ 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 8, 2],
                         [ 5, 0, 3, 6, 0, 0, 6, 0, 0, 9, 0, 5, 7, 4],
                         [10,10, 8, 0, 0, 0, 0, 3, 6, 0, 0, 0, 2, 8],
                         [ 0,12, 0, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0,10],
                         [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,10, 0, 0, 0]])
city_sell = {'anafola':['raw fish', 'cooked fish', 'string', 'beads', 'sand', 'scales', 'bark', 'lead', 'tin', 'copper',' iron', 'persuasion book'],
             'benfriege':['raw fish', 'cooked fish', 'well cooked fish', 'string', 'beads', 'scales', 'bark', 'critical thinking book', 'crafting book', 'survival book', 'gathering book'],
             'demetry':['raw meat', 'cooked meat', 'fruit', 'string', 'beads', 'hide', 'sand', 'clay', 'leather', 'ceramic', 'glass', 'gems', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'tungsten', 'bartering book', 'crafting book'],
             'enfeir':['raw meat', 'cooked meat', 'string', 'hide', 'tin', 'copper', 'aluminum', 'kevlium'],
             'fodker':['raw meat', 'cooked meat', 'string', 'hide', 'sand', 'lead', 'tin', 'copper', 'iron', 'excavating book'],
             'glaser':['raw fish', 'cooked fish', 'string', 'beads', 'scales', 'bark', 'lead', 'tin', 'critical thinking book', 'survival book', 'gathering book'],
             'kubani':['raw meat', 'string', 'beads', 'sand', 'clay', 'glass', 'lead', 'copper', 'iron', 'tantalum', 'titanium', 'survival book'],
             'pafiz':['cooked meat', 'cooked fish', 'fruit', 'string', 'beads', 'hide', 'scales', 'iron', 'nickel', 'persuasion book'],
             'scetcher':['raw meat', 'raw fish', 'string', 'sand', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'aluminum', 'kevlium', 'nickel'],
             'starfex':['raw fish', 'cooked fish', 'fruit', 'string', 'beads', 'scales', 'lead', 'tantalum', 'tungsten', 'heating book'],
             'tamarania':['raw meat', 'cooked meat', 'well cooked meat', 'string', 'beads', 'hide', 'clay', 'leather', 'lead', 'tin', 'copper', 'iron', 'tantalum', 'aluminum', 'kevlium', 'nickel', 'titanium', 'diamond', 'smithing book'],
             'tamariza':['fruit', 'string', 'beads', 'bark', 'rubber', 'iron', 'nickel', 'chromium', 'heating book'],
             'tutalu':['raw meat', 'cooked meat', 'string', 'beads', 'hide', 'leather', 'copper', 'kevlium', 'diamond', 'excavating book'],
             'zinzibar':['raw meat', 'cooked meat', 'string', 'hide', 'lead', 'tin', 'tantalum', 'aluminum']}
city_labor = {'anafola':{'Persuasion':5, 'Excavating':5},
             'benfriege':{'Critical Thinking':5, 'Persuasion':5, 'Crafting':8, 'Survival':5},
             'demetry':{'Bartering':8, 'Crafting':5},
             'enfeir':{'Critical Thinking':5, 'Heating':5, 'Smithing':5, 'Stealth':8},
             'fodker':{'Bartering':5, 'Smithing':5},
             'glaser':{'Critical Thinking':5, 'Persuasion':5, 'Crafting':5, 'Survival':5, 'Excavating':5},
             'kubani':{'Critical Thinking':5, 'Bartering':5, 'Crafting':5, 'Gathering':5},
             'pafiz':{'Persuasion':8, 'Crafting':5, 'Heating':5, 'Gathering':5},
             'scetcher':{'Smithing':5, 'Stealth':5},
             'starfex':{'Heating':5, 'Gathering':5, 'Excavating':5},
             'tamarania':{'Smithing':8},
             'tamariza':{'Critical Thinking':5, 'Persuasion':5, 'Heating':5},
             'tutalu':{'Smtihing':5, 'Excavating':5},
             'zinzibar':{'Persuasion':5, 'Smithing':5, 'Survival':8}}
skills = ['Persuasion', 'Critical Thinking', 'Heating', 'Survival', 'Smithing', 'Crafting', 'Excavating', 'Stealth', 'Gathering', 'Bartering']
titles = {'explorer': {'titleVP': 5, 'minTitleReq': 20, 'value':0, 'holder': None, 'record': None, 'category': 'General', 'description': 'Most unique tiles traveled upon.'},
          'loyal': {'titleVP': 2, 'minTitleReq': 25, 'value':0, 'holder': None, 'record': None, 'category': 'General', 'in_birthcity':True, 'description': 'Most rounds spent in their birth city (all actions per round).'},
          'valiant': {'titleVP': 3, 'minTitleReq': 6, 'value':0, 'holder': None, 'record': None, 'category': 'Combat', 'description': 'Maximum difference between an opponent stronger than you, that you defeated, and your total combat at the start of battle.'},
          'sorcerer': {'titleVP': 5, 'minTitleReq': 2, 'value':0, 'holder': None, 'record': None, 'category': 'Fellowship', 'description': 'Longest continuous Tamariza Wizard Tower premium holder.'},
          'superprime': {'titleVP': 5, 'minTitleReq': 40, 'value':0, 'holder': None, 'record': None, 'category': 'Fellowship', 'description': 'Highest credit score from the Demetry Grand Bank.'},
          'traveler': {'titleVP': 3, 'minTitleReq': 30, 'value':0, 'holder': None, 'record': None, 'category': 'General', 'description': 'Most road tile movement.'},
          'apprentice': {'titleVP': 4, 'minTitleReq': 60, 'value':0, 'holder': None, 'record': None, 'category': 'Knowledge', 'description': 'Most cumulative levels gained from trainers.'},
          'scholar': {'titleVP': 5, 'minTitleReq': 5, 'value':0, 'holder': None, 'record': None, 'category': 'Fellowship', 'description': 'Most books checked out from the Benefriege public library, learned from, and returned.'},
          'laborer': {'titleVP': 3, 'minTitleReq': 10, 'value':0, 'holder': None, 'record': None, 'category': 'Capital', 'description': 'Most jobs worked.'},
          'valuable': {'titleVP': 2, 'minTitleReq': 20, 'value':0, 'holder': None, 'record': None, 'category': 'Capital', 'description': 'Most money earned from jobs worked.'},
          'entrepreneur': {'titleVP': 3, 'minTitleReq': 50, 'value':0, 'holder': None, 'record': None, 'category': 'Capital', 'description': 'Most money received from owned markets.'},
          'trader': {'titleVP': 4, 'minTitleReq': 5, 'value':0, 'holder': None, 'record': None, 'category': 'Capital', 'unique_traders': set(), 'description': 'Most trading (buy or sell) with unique traders.'},
          'negotiator': {'titleVP': 4, 'minTitleReq': 15, 'value':0, 'holder': None, 'record': None, 'category': 'Knowledge', 'description': 'Most successful persuasions.'},
          'steady': {'titleVP': 4, 'minTitleReq': 15, 'value':0, 'holder': None, 'record': None, 'category': 'General', 'description': 'Most rounds started with zero fatigue.'},
          'grinder': {'titleVP': 3, 'minTitleReq': 120, 'value':0, 'holder': None, 'record': None, 'category': 'General', 'description': 'Most fatigue taken.'},
          'merchant': {'titleVP': 2, 'minTitleReq': 40, 'value':0, 'holder': None, 'record': None, 'category': 'Capital', 'description': 'Most coin made from selling items to merchants or traders.'},
          'brave': {'titleVP': 5, 'minTitleReq': 80, 'value':0, 'holder': None, 'record': None, 'category': 'Combat', 'currentStreak': 0, 'description': 'Sum total level of continuous enemies defeated without restoring HP.'},
          'decisive': {'titleVP': 1, 'minTitleReq': -float('inf'), 'holder': None, 'record': None, 'category': 'General', 'value':-float('inf'), 'sum': 0, 'round': 0, 'description': 'Shortest average round time.'},
          'champion': {'titleVP': 7, 'minTitleReq': 1, 'value':0, 'holder': None, 'record': None, 'category': 'Fellowship', 'description': 'Most class V Scetcher Tournament titles won.'},
          'resurrector': {'titleVP': 10, 'minTitleReq': 1, 'value':0, 'holder': None, 'record': None, 'category': 'End Game', 'description': 'Strongest magic sword revived with its ability unlocked.'}}

records = {}


def conn2set():
    P = [np.concatenate((connectivity.T[:i, i], connectivity.T[i, i:])) for i in range(len(connectivity))]
    S = {}
    for i in range(len(P)):
        for j in range(len(P[i])):
            if P[i][j] > 0:
                S[frozenset([cities[i], cities[j]])] = P[i][j]
    return S
Skirmishes = [conn2set(), {}]
def getSkirmish():
    for S in Skirmishes[0]:
        if S in Skirmishes[1]:
            continue
        if np.random.rand() < (1 / (Skirmishes[0][S] + 1)):
            Skirmishes[1][S] = 3
    popS = []
    for S in Skirmishes[1]:
        Skirmishes[1][S] -= 1
        if Skirmishes[1][S] <= 0:
            popS.append(S)
    for S in popS:
        Skirmishes[1].pop(S)
def getTodaysMarket(max_items=6):
    city_markets, S = {}, []
    for city in city_sell:
        city_markets[city] = set(np.random.choice(city_sell[city], max_items))
        S.append(f'{city},'+','.join(list(city_markets[city])))
    return '|'.join(S)
def rbtwn(mn, mx):
    return np.random.choice(np.arange(mn, mx+1))
def getTodaysJobs():
    city_jobs = {}
    for city in city_labor:
        s = set()
        for skill in skills:
            val = city_labor[city][skill] if skill in city_labor[city] else 2
            if rbtwn(1, 10) <= val:
                s.add(skill)
        city_jobs[city] = s
    return city_jobs
            

def updateServer(username, category, msg):
    global Skirmishes, records
    if category == '[LAUNCH]':
        if msg == 'Listening':
            if len(client_gameStatus) == 1:
                difficulty[0] = 'moderate'
                seed[0] = np.random.randint(1, 100000)
                dtm = str(datetime.datetime.now())
                dtm = dtm[:dtm.index('.')].replace(':','').replace(' ','_') # Exclude the milliseconds
                default_save[0] = dtm + f'_{seed[0]}'
                default_load[0] = 'None'
                gameEnd[0] = '2:100'
            client_gameStatus[username]['ready'] = False
            client_gameStatus[username]['round end'] = False
            client_gameStatus[username]['end'] = False
            for other_username in client_gameStatus:
                if other_username != username:
                    send_msg = 'Ready' if client_gameStatus[other_username]['ready'] else 'Not Ready'
                    sendMessage(other_username, username, '[LAUNCH]', send_msg)
                else:
                    sendMessage('[SERVER]', username, '[DIFFICULTY]', difficulty[0])
                    sendMessage('[SERVER]', username, '[SEED]', seed[0])
                    sendMessage('[SERVER]', username, '[SAVE]', default_save[0])
                    sendMessage('[SERVER]', username, '[LOAD]', default_load[0])
                    sendMessage('[SERVER]', username, '[END SETTING]', gameEnd[0])
        elif msg == 'Ready':
            client_gameStatus[username]['ready'] = True
            client_gameStatus[username]['round end'] = False
            all_ready = True
            for D in client_gameStatus.values():
                if D['ready'] == False:
                    all_ready = False
                    break
            if all_ready:
                game_launched[0] = True
                # initiate records
                for title in titles:
                    record_data = [[username, titles[title]['value']] for username in client_gameStatus.keys()]
                    rd = pd.DataFrame(record_data, columns=['username', 'value']).set_index('username')
                    records[title] = rd
        elif msg == 'Not Ready':
            client_gameStatus[username]['ready'] = False
    elif category == '[CLAIM]':
        client_gameStatus[username]['birth city'] = msg
    elif category == '[DIFFICULTY]':
        difficulty[0] = msg
    elif category == '[SEED]':
        seed[0] = msg
    elif category == '[SAVE]':
        default_save[0] = msg
    elif category == '[LOAD]':
        default_load[0] = msg
    elif category == '[LOAD SKIRMISHES]':
        Skirmishes = msg
    elif category == '[END SETTING]':
        gameEnd[0] = msg
    elif (category == '[ROUND]') and (msg == 'end'):
        client_gameStatus[username]['round end'] = True
        all_ended = True
        for D in client_gameStatus.values():
            if ('round end' in D) and (D['round end'] == False):
                all_ended = False
                break
        if all_ended:
            all_ended = False
            getSkirmish()
            market_seed = np.random.randint(100000)
            for username in client_gameStatus:
                if 'round end' in client_gameStatus[username]: 
                    client_gameStatus[username]['round end'] = False
                sendMessage('[SERVER]', username, '[SKIRMISH]', Skirmishes[1])
                # Give the same random market seed to each player to generate jobs and markets
                sendMessage('[SERVER]', username, '[MARKET]', market_seed)
    elif category == '[TITLE VALUE]':
        title = msg['title']
        # assume the msg is a dict with keys {'title', 'username', 'value'}
        records[title].loc[msg['username'], 'value'] = msg['value']
        # Now see who the max record holder is
        mx = records[msg['title']].sort_values('value').iloc[-1]
        new_record = {'value': mx.value, 'holder': mx.name, 'title': title}
        if mx.value >= titles[title]['minTitleReq']:
            # Check if new max is greater than previous holder
            if (titles[title]['holder'] is not None) and (mx.name == titles[title]['holder']):
                sendMessage('[SERVER]', username, '[TITLE CHANGE]', new_record)
            elif (titles[title]['holder'] is not None) and (mx.value > records[title].loc[titles[title]['holder'], 'value']):
                # This handles both cases when new player goes beyond previous holder and also when previous holder loses value to someone else who deserves it.
                sendMessage('[SERVER]', username, '[TITLE CHANGE]', new_record)
            elif titles[title]['holder'] is None:
                sendMessage('[SERVER]', username, '[TITLE CHANGE]', new_record)
        elif titles[title]['holder'] is not None:
            # This means that whoever the holder was has lost value and now no one should be the record holder.
            sendMessage('[SERVER]', username, '[TITLE CHANGE]', {'value': -float('inf'), 'holder': None, 'title': title})
    elif category == '[REDUCED TENSION]':
        Skirmishes[0][msg[0]] += msg[1]
    elif category == '[END STATS]':
        client_gameStatus[username]['end'] = msg
        all_sent = True
        stats = {}
        for username, D in client_gameStatus.items():
            if ('round end' in D) and (D['end'] == False):
                all_sent = False
                break
            else:
                stats[username] = D['end']
        if all_sent:
            for username in client_gameStatus:
                sendMessage('[SERVER]', username, '[FINAL END STATS]', stats)

def close_socket(notified_socket):
    closed_username = clients[notified_socket]['data'].decode('utf-8')
    print('Closed connection from: {}'.format(closed_username))

    # Remove from list for socket.socket()
    sockets_list.remove(notified_socket)

    # Remove from our list of users
    _ = clients.pop(notified_socket)
    del _
    _ = client_code.pop(closed_username)
    del _
    _ = client_gameStatus.pop(closed_username)
    del _
    
    if len(client_code) == 0:
        game_launched[0] = False
        difficulty[0] = None
        seed[0] = None
        default_save[0] = None
        default_load[0] = None
        Skirmishes[0] = conn2set()
        print("All connections closed. Resetting Stats.")
    else:
        for username in client_code:
            sendMessage(closed_username, username, '[CONNECTION]', 'Closed') 

def launch_server():
    Thread(target=listen, args=(), daemon=True).start()

def listen():    
    while True:
    
        # Calls Unix select() system call or Windows select() WinSock call with three parameters:
        #   - rlist - sockets to be monitored for incoming data
        #   - wlist - sockets for data to be send to (checks if for example buffers are not full and socket is ready to send some data)
        #   - xlist - sockets to be monitored for exceptions (we want to monitor all sockets for errors, so we can use rlist)
        # Returns lists:
        #   - reading - sockets we received some data on (that way we don't have to check sockets manually)
        #   - writing - sockets ready for data to be send thru them
        #   - errors  - sockets with some exceptions
        # This is a blocking call, code execution will "wait" here and "get" notified in case any action should be taken
        read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    
    
        # Iterate over notified sockets
        for notified_socket in read_sockets:
    
            # If notified socket is a server socket - new connection, accept it
            if notified_socket == server_socket:
    
                # Accept new connection
                # That gives us new socket - client socket, connected to this given client only, it's unique for that client
                # The other returned object is ip/port set
                client_socket, client_address = server_socket.accept()
    
                # Client should send his name right away, receive it
                user = receive_message(client_socket)
    
                # If False - client disconnected before he sent his name
                if user is False:
                    continue
                username = user['data'].decode('utf-8')
                if username in client_code:
                    print(f"Duplicate username: {username} attempted to join. Connection Rejected.")
                    continue
    
                # Add accepted socket to select.select() list
                sockets_list.append(client_socket)
    
                # Also save username and username header
                clients[client_socket] = user
                
                client_code[username] = {'user':user, 'socket':client_socket}
                client_gameStatus[username] = {}
    
                print('Accepted new connection from {}:{}, username: {}'.format(*client_address, user['data'].decode('utf-8')))
    
            # Else existing socket is sending a message
            else:
    
                # Receive message
                message = receive_message(notified_socket)
    
                # If False, client disconnected, cleanup
                if message is False:
                    close_socket(notified_socket)
                    continue
    
                # Get user by notified socket, so we will know who sent the message
                user = clients[notified_socket]
                username, category, msg = decoded_message(user, message)
    
                print(f'Received message from {username}: {category} {msg}')
                
                updateServer(username, category, msg)
                    
                # Iterate over connected clients and broadcast message
                for client_socket in clients:
    
                    # But don't sent it to sender
                    if client_socket != notified_socket:
    
                        # Send user and message (both with their headers)
                        # We are reusing here message header sent by sender, and saved username header send by user when he connected
                        client_socket.send(user['2nd header'] + user['data'] + message['header'] + message['category'] + message['2nd header'] + message['data'])
    
        # It's not really necessary to have this, but will handle some socket exceptions just in case
        for notified_socket in exception_sockets:
            close_socket(notified_socket)
            
if __name__ == "__main__":
    listen()