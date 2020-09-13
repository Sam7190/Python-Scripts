import socket
import errno
import pickle
from threading import Thread

HEADER_LENGTH = 20
client_socket = None
joinCategory = '[CONNECTION]'

# Connects to the server
def connect(ip, port, my_username, error_callback):

    global client_socket

    # Create a socket
    # socket.AF_INET - address family, IPv4, some otehr possible are AF_INET6, AF_BLUETOOTH, AF_UNIX
    # socket.SOCK_STREAM - TCP, conection-based, socket.SOCK_DGRAM - UDP, connectionless, datagrams, socket.SOCK_RAW - raw IP packets
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to a given ip and port
        client_socket.connect((ip, port))
    except Exception as e:
        # Connection error
        error_callback('Connection error: {}'.format(str(e)))
        return False

    # Prepare username and header and send them
    # We need to encode username to bytes, then count number of bytes and prepare header of fixed size, that we encode to bytes as well
    username = my_username.encode('utf-8')
    username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
    category = joinCategory.encode('utf-8')
    category_header = f"{len(category):<{HEADER_LENGTH}}".encode('utf-8')
    client_socket.send(category_header + category + username_header + username)

    return True

# Sends a message to the server
def send(category, message):
    # Encode message to bytes, prepare header and convert to bytes, like for username above, then send
    category = category.encode('utf-8')
    category_header = f"{len(category):<{HEADER_LENGTH}}".encode('utf-8')
    if type(message) is str:
        message = message.encode('utf-8')
    else:
        message = pickle.dumps(message)
    message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
    client_socket.send(category_header + category + message_header + message)

# Starts listening function in a thread
# incoming_message_callback - callback to be called when new message arrives
# error_callback - callback to be called on error
def start_listening(incoming_message_callback, error_callback):
    Thread(target=listen, args=(incoming_message_callback, error_callback), daemon=True).start()

# Listens for incomming messages
def listen(incoming_message_callback, error_callback):
    while True:

        try:
            # Now we want to loop over received messages (there might be more than one) and print them
            while True:

                # Receive our "header" containing username length, it's size is defined and constant
                username_header = client_socket.recv(HEADER_LENGTH)

                # If we received no data, server gracefully closed a connection, for example using socket.close() or socket.shutdown(socket.SHUT_RDWR)
                if not len(username_header):
                    error_callback('Connection closed by the server')

                # Convert header to int value
                username_length = int(username_header.decode('utf-8').strip())

                # Receive and decode username
                username = client_socket.recv(username_length).decode('utf-8')
                print(username)
                category_header = client_socket.recv(HEADER_LENGTH)
                category_length = int(category_header.decode('utf-8').strip())
                category = client_socket.recv(category_length).decode('utf-8')
                print(category)

                # Now do the same for message (as we received username, we received whole message, there's no need to check if it has any length)
                message_header = client_socket.recv(HEADER_LENGTH)
                message_length = int(message_header.decode('utf-8').strip())
                receivedMessage = client_socket.recv(message_length)
                print(receivedMessage)
                try:
                    message = receivedMessage.decode('utf-8')
                except UnicodeDecodeError:
                    message = pickle.loads(receivedMessage)
                # Print message
                incoming_message_callback(username, category, message)

        except Exception as e:
            # Any other exception - something happened, exit
            error_callback('Reading error: {}'.format(str(e)))