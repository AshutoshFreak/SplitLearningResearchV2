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
from utils.connections import get_object
from concurrent.futures import ThreadPoolExecutor
from utils.merge_grads import merge_grads
import torch.optim as optim
import multiprocessing
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
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


def main(server_pipe_endpoints, args):
    HOST = 'localhost'
    PORT = 8000
    accept_clients = AcceptClients(HOST, PORT, args.number_of_clients)
    # accept_clients.start()
    ServerSocket = accept_clients.run(server_pipe_endpoints)
    connected_client_ids = list(connected_clients.keys())
    # time.sleep(10)
    # accept_clients.stop()

    # print(len(connected_clients))

    for client_id in connected_clients:
        client = connected_clients[client_id]
        client.front_model = MNIST_CNN.front()
        client.back_model = MNIST_CNN.back()
        client.center_model = MNIST_CNN.center()
        # client.train_fun = MNIST_CNN.train
        # client.test_fun = MNIST_CNN.test


    for client_id in connected_clients:
        client = connected_clients[client_id]

        # Attaching PrivacyEngine to center model
        # initialize privacy accountant
        # client.center_accountant = RDPAccountant()

        # wrap model
        #####################################
        # client.center_model = GradSampleModule(client.center_model)
        client.center_model.to(client.device)

        # initialize optimizer
        client.center_optimizer = optim.SGD(client.center_model.parameters(), lr=0.05)
        
        
        #####################################
        # # wrap optimizer
        # client.center_optimizer = DPOptimizer(
        #     optimizer=client.center_optimizer,
        #     # noise_multiplier=noise_multiplier, # same as make_private arguments
        #     max_grad_norm=1.0, # same as make_private arguments
        #     expected_batch_size=args.batch_size # if you're averaging your gradients, you need to know the denominator
        # )

        #####################################
        # # attach accountant to track privacy for an optimizer
        # client.center_optimizer.attach_step_hook(
        #     client.center_accountant.get_optimizer_hook_fn(
        #     # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
        #     sample_rate=args.batch_size/60000
        #     )
        # )


    with ThreadPoolExecutor() as executor:
        for _, client in connected_clients.items():
            executor.submit(client.send_model())


        if args.server_side_tuning:
            # [Server side tuning]
            # 1 client is kept as a dummy client and is stripped from 'clients' dict
            # dummy client is the first client generated by generate_random_clients
            # *edge cases not considered. num_clients should be > 1*
            dummy_client_id = connected_client_ids[0]
            connected_client_ids = connected_client_ids[1:]
            dummy_client = connected_clients[dummy_client_id]
            connected_clients.pop(dummy_client_id)
        
        first_client = connected_clients[connected_client_ids[0]]
        num_iterations, num_test_iterations = get_object(first_client.conn)

        print('server.py')
        print(connected_client_ids)
        # Training
        for epoch in range(args.epochs):
            print(f'\nEpoch: {epoch+1}')
            for iteration in range(num_iterations):
                for _, client in connected_clients.items():
                    executor.submit(client.get_remote_activations1())


                for _, client in connected_clients.items():
                    executor.submit(client.forward_center())


                for _, client in connected_clients.items():
                    executor.submit(client.send_remote_activations2())


                for _, client in connected_clients.items():
                    executor.submit(client.get_remote_activations2_grads())
                    executor.submit(client.backward_center())

                for _, client in connected_clients.items():
                    executor.submit(client.send_remote_activations1_grads())

                params = []
                for _, client in connected_clients.items():
                    params.append(client.center_model.parameters())
                merge_grads(params)


                for _, client in connected_clients.items():
                    client.center_optimizer.step()


                for _, client in connected_clients.items():
                    client.center_optimizer.zero_grad()

            ##########################################
            # epsilon, best_alpha = client.center_accountant.get_privacy_spent(delta=delta)
            # print(f"([server] ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}")

            if args.server_side_tuning:
                # [server side tuning]
                print('Server side tuning')
                for iteration in range(num_iterations):
                    dummy_client.get_remote_activations1()
                    dummy_client.forward_center()
                    dummy_client.send_remote_activations2()
                    dummy_client.get_remote_activations2_grads()
                    dummy_client.backward_center()
                    dummy_client.send_remote_activations1_grads()

                    # params = []
                    # params.append(client.center_model.parameters())
                    # merge_grads(params)

                    dummy_client.center_optimizer.step()
                    dummy_client.center_optimizer.zero_grad()


            # Testing
            with torch.no_grad():
                for iteration in range(num_test_iterations):
                    for _, client in connected_clients.items():
                        executor.submit(client.get_remote_activations1())

                    for _, client in connected_clients.items():
                        executor.submit(client.forward_center())

                    for _, client in connected_clients.items():
                        executor.submit(client.send_remote_activations2())


if __name__ == "__main__":
    main(None, None)