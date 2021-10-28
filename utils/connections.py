import logging
# import socket
import pickle
import time

# def is_socket_closed(sock: socket.socket) -> bool:
#         logger = logging.getLogger(__name__)
#         try:
#             # this will try to read bytes without blocking without removing them from buffer (peek only)
#             data = sock.recv(16, socket.MSG_DONTWAIT | socket.MSG_PEEK)
#             if len(data) == 0:
#                 return True
#         except BlockingIOError:
#             return False  # socket is open and reading from it would block
#         except ConnectionResetError:
#             return True  # socket was closed for some other reason
#         except Exception:
#             logger.exception("unexpected exception when checking if a socket is closed")
#             return False
#         return False 


def send_object(socket, data):
    with open(f'{socket}/barrier', 'rb') as bar:
        barrier = pickle.load(bar)

    barrier.close_it()

    with open(f'{socket}/barrier', 'wb') as bar:
        pickle.dump(barrier, bar)

    with open(f'{socket}/activ_grad', 'wb') as activ_grad:
        pickle.dump(data, activ_grad)

    barrier.open_it()
    
    with open(f'{socket}/barrier', 'wb') as bar:
        pickle.dump(barrier, bar)

    # HEADERSIZE = 10
    # data = bytes(f'{len(data):<{HEADERSIZE}}', 'utf-8') + data
    # socket.send(data)



def get_object(socket):
    # HEADERSIZE = 10
    # full_data = b''
    # new_data = True
    # while True:
    #     data = socket.recv(4096)
    #     if new_data:
    #         datalen = int(data[:HEADERSIZE])
    #         new_data = False
        
    #     full_data += data
        
    #     if len(full_data) - HEADERSIZE == datalen:
    #         d = pickle.loads(full_data[HEADERSIZE:])
    #         break
    
    # return d

    with open(f'{socket}/barrier', 'rb') as bar:
        barrier = pickle.load(bar)

    while(barrier.is_closed()):
        time.sleep(0.1)
        with open(f'{socket}/barrier', 'rb') as bar:
            barrier = pickle.load(bar)
    
    with open(f'{socket}/activ_grad', 'rb') as activ_grad:
        data = pickle.load(activ_grad)
        barrier.close_it()
        with open(f'{socket}/barrier', 'wb') as bar:
            pickle.dump(barrier, bar)

