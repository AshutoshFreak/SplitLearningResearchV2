import logging
import socket
import pickle

def is_socket_closed(sock: socket.socket) -> bool:
        logger = logging.getLogger(__name__)
        try:
            # this will try to read bytes without blocking without removing them from buffer (peek only)
            data = sock.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
            if len(data) == 0:
                return True
        except BlockingIOError:
            return False  # socket is open and reading from it would block
        except ConnectionResetError:
            return True  # socket was closed for some other reason
        except Exception:
            logger.exception("unexpected exception when checking if a socket is closed")
            return False
        return False 


def send_object(socket, data):
    # data = pickle.dumps(data)
    # HEADERSIZE = 10
    # data = bytes(f'{len(data):<{HEADERSIZE}}', 'utf-8') + data
    socket.send(data)


def get_object(socket):
    # HEADERSIZE = 10
    # full_data = b''
    # new_data = True
    data = socket.recv()
    # while not data:
    #     data = socket.recv()
        # if new_data:
        #     datalen = int(data[:HEADERSIZE])
        #     new_data = False
        
        # full_data += data
        
        # if len(full_data) - HEADERSIZE == datalen:
        #     d = pickle.loads(full_data[HEADERSIZE:])
        #     break
    
    return data
    
