import socket
from _thread import start_new_thread
from threading import Thread
from ConnectedClient import ConnectedClient
import os
import time
import pickle
import torch
import random
from models import MNIST_CNN
from concurrent.futures import ThreadPoolExecutor
from utils.merge_grads import merge_grads
import torch.optim as optim
import multiprocessing
# import requests

SEED = 2646
random.seed(SEED)
torch.manual_seed(SEED)

ThreadCount = 0

def increaseThreadCount():
    global ThreadCount
    ThreadCount += 1


def decreaseThreadCount():
    global ThreadCount
    ThreadCount -= 1


def getThreadCount():
    return ThreadCount

# def threaded_client(connection):
#     data = connection.recv(4096)
#     if not data:
#         print(f'\n[*] Disconnected from {connection.getpeername()[0]}:{connection.getpeername()[1]}')
#         connection.close()
#         decreaseThreadCount()
#         print(f'Total clients connected: {getThreadCount()}')w
#         break
#     reply = ''
#     try:
#         target, message = data.split('/')
#         ip, port = target.split(':')
        
#         port = int(port)
#         target = (ip, port)
#         if target not in mapping.keys():
#             reply = '[*] Server Error: Cannot find specified target client'
#         else:
#             source = connection.getpeername()
#     except:
#         reply = '[*] Parse Error: Invalid Input Format'

#     if reply != '':
#         connection.sendall(reply)

connected_clients = {}
class AcceptClients(Thread):
    def __init__(self, host, port, limit):
        Thread.__init__(self)
        self.host = host
        self.port = port
        # self.keepRunning = True
        self.limit = limit
    

    def run(self, server_pipe_endpoints):
        ServerSocket = multiprocessing.Pipe()
        for client_id in server_pipe_endpoints:
            conn = server_pipe_endpoints[client_id]
            connected_clients[client_id] = ConnectedClient(client_id, conn)
            print(f'\n[*] Connected to: {client_id}')
            increaseThreadCount()
            print(f'Total clients connected: {ThreadCount}')
        
        return ServerSocket
        # self.keepRunning = False

        # ServerSocket.close()
        # print('socket closed')


    # def stop(self):
    #     self.keepRunning = False


def main(server_pipe_endpoints):
    num_epochs = 10
    HOST = 'localhost'
    PORT = 8000
    limit_clients = 2
    accept_clients = AcceptClients(HOST, PORT, limit_clients)
    # accept_clients.start()
    ServerSocket = accept_clients.run(server_pipe_endpoints)
    # time.sleep(10)
    # accept_clients.stop()

    # print(len(connected_clients))

    for client_id in connected_clients:
        client = connected_clients[client_id]
        client.front_model = MNIST_CNN.front()
        client.back_model = MNIST_CNN.back()
        client.center_model = MNIST_CNN.center().to(client.device)
        client.train_fun = MNIST_CNN.train
        client.test_fun = MNIST_CNN.test
        client.center_optimizer = optim.Adadelta(client.center_model.parameters(), lr=1)


    with ThreadPoolExecutor() as executor:
        for _, client in connected_clients.items():
            executor.submit(client.send_model())
        
        # Training
        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch+1}')
            for _, client in connected_clients.items():
                executor.submit(client.get_remote_activations1())


            for _, client in connected_clients.items():
                executor.submit(client.forward_center())


            for _, client in connected_clients.items():
                executor.submit(client.send_remote_activations2())


            for _, client in connected_clients.items():
                executor.submit(client.get_remote_activations2_grads())


            for _, client in connected_clients.items():
                client.center_optimizer.zero_grad()
                executor.submit(client.backward_center())

            for _, client in connected_clients.items():
                executor.submit(client.send_remote_activations1_grads())

            params = []
            for _, client in connected_clients.items():
                params.append(client.center_model.parameters())
            merge_grads(params)

            for _, client in connected_clients.items():
                client.center_optimizer.step()

        # Testing
        for _, client in connected_clients.items():
            executor.submit(client.get_remote_activations1())

        for _, client in connected_clients.items():
            executor.submit(client.forward_center())

        for _, client in connected_clients.items():
            executor.submit(client.send_remote_activations2())



if __name__ == "__main__":
    main(None)