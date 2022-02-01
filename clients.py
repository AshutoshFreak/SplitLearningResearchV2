import os
import random
import string
import socket
import requests
import sys
import threading
import time
import torch
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor as Executor
from utils.split_dataset import split_dataset
from utils.client_simulation import generate_random_clients
import matplotlib.pyplot as plt
import time
import server
import multiprocessing


# should get optimizers from the server instead of initializing here
# will do later
import torch.optim as optim


SEED = 2647
random.seed(SEED)
torch.manual_seed(SEED)


def initialize_client(client, dataset, train_batch_size, test_batch_size, transform=None):
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(train_batch_size, test_batch_size)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    num_clients = 2
    num_epochs = 10

    server_pipe_endpoints = {}

    dataset = '1000Fundus'
    print(f'Using dataset: {dataset}')
    train_batch_size = 8
    test_batch_size = 8

    time_taken = {'forward_front':0,
                    'send_remote_activations1':0,
                    'get_remote_activations2':0,
                    'forward_back':0,
                    'calculate_loss':0,
                    'calculate_train_acc':0,
                    'zero_grad':0,
                    'backward_back':0,
                    'send_remote_activations2_grads':0,
                    'get_remote_activations1_grads':0,
                    'backward_front':0,
                    'step':0
                }

    overall_acc = []
    print('Generating random clients...', end='')
    clients = generate_random_clients(num_clients)
    print('Done')

    client_ids = clients.keys()
    print(f'Random client ids:{str(list(client_ids))}')
    
    print('Splitting dataset...', end='')
    split_dataset(dataset, list(client_ids))
    print('Done')


    # executor.submit will enable each function to run in separate threads for each client
    with Executor() as executor:
        # all clients load data and create dataloaders
        print('Initializing clients...')
        for _, client in clients.items():
            executor.submit(initialize_client(client, dataset, train_batch_size, test_batch_size))
        print('Client Intialization complete.')


        # all clients connect to the server
        for _, client in clients.items():
            executor.submit(client.connect_server())
        
        for client_id in clients:
            server_pipe_endpoints[client_id] = clients[client_id].server_socket
        
        # start server and provide pipe endpoints
        p = multiprocessing.Process(target=server.main, args=(server_pipe_endpoints,))
        p.start()
        

        # all clients get model from the server
        print('Getting model from server...', end='')
        for _, client in clients.items():
            executor.submit(client.get_model())
        print('Done')

        for _, client in clients.items():
            print(client.front_model)
            print(client.back_model)

        for _, client in clients.items():
            client.front_model.to(client.device)
            client.back_model.to(client.device)
            # client.front_optimizer = optim.Adadelta(client.front_model.parameters(), lr=0.1)
            client.back_optimizer = optim.AdamW(client.back_model.parameters(), lr=1e-3)
            client.scheduler = optim.lr_scheduler.StepLR(client.back_optimizer, step_size=2, gamma=0.9,verbose=True)


        # Training
        num_iters = 50
        for epoch in range(num_epochs):
            for _, client in clients.items():
                client.iterator = iter(client.train_DataLoader)
            
            print(f"Epoch: {epoch}")
            
            for iter_num in range(num_iters):
                print(f'Epoch: {epoch}, Iter: {iter_num+1}:')

                # call forward prop for each client         
                for _, client in clients.items():
                    executor.submit(client.forward_front(transferlearning=True))


                # send activations to the server
                for _, client in clients.items():
                    executor.submit(client.send_remote_activations1())


                for _, client in clients.items():
                    executor.submit(client.get_remote_activations2())


                for _, client in clients.items():
                    executor.submit(client.forward_back())


                for _, client in clients.items():
                    executor.submit(client.calculate_loss())


                for _, client in clients.items():
                    executor.submit(client.calculate_train_acc())


                for _, client in clients.items():
                    executor.submit(client.zero_grad())


                for _, client in clients.items():
                    executor.submit(client.backward_back())


                # start = time.time()
                # for _, client in clients.items():
                #     executor.submit(client.send_remote_activations2_grads())
                # end = time.time()
                # time_taken['send_remote_activations2_grads'] += end-start


                # start = time.time()
                # for _, client in clients.items():
                #     executor.submit(client.get_remote_activations1_grads())
                # end = time.time()
                # time_taken['get_remote_activations1_grads'] += end-start


                # start = time.time()
                # for _, client in clients.items():
                #     executor.submit(client.backward_front())
                # end = time.time()
                # time_taken['backward_front'] += end-start


                for _, client in clients.items():
                    executor.submit(client.step())
            
            for _, client in clients.items():
                client.scheduler.step()


            train_acc = 0
            for _, client in clients.items():
                train_acc += client.train_acc[-1]
            train_acc = train_acc/num_clients
            overall_acc.append(train_acc)


        # # Testing
        # # Setting up iterator for testing
        # for _, client in clients.items():
        #     client.iterator = iter(client.test_DataLoader)

        # # call forward prop for each client
        # for _, client in clients.items():
        #     executor.submit(client.forward_front())

        # # send activations to the server
        # for _, client in clients.items():
        #     executor.submit(client.send_remote_activations1())

        # for _, client in clients.items():
        #     executor.submit(client.get_remote_activations2())

        # for _, client in clients.items():
        #     executor.submit(client.forward_back())

        # for _, client in clients.items():
        #     executor.submit(client.calculate_loss())

            # for _, client in clients.items():
            #     executor.submit(client.calculate_test_acc())


    for client_id, client in clients.items():
        plt.plot(list(range(num_epochs)), client.train_acc, label=f'{client_id} (Max:{max(client.train_acc):.4f})')
    plt.plot(list(range(num_epochs)), overall_acc, label=f'Average (Max:{max(overall_acc):.4f})')
    plt.title(f'{num_clients} Clients: Train Accuracy vs. Epochs')
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f'./results/train_acc_vs_epoch/{num_clients}_clients_{num_epochs}_epochs.png', bbox_inches='tight')
    plt.show()


    print('Test accuracy for each client:')
    for client_id, client in clients.items():
            print(f'{client_id}:{client.test_acc}')
