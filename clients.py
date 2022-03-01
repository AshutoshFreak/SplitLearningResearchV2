import os
import random
import string
import socket
import tqdm
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
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator


# should get optimizers from the server instead of initializing here
# will do it later
import torch.optim as optim


SEED = 2647
random.seed(SEED)
torch.manual_seed(SEED)


def initialize_client(client, dataset, train_batch_size, test_batch_size, tranform=None):
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(train_batch_size, test_batch_size)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    num_clients = 1
    num_epochs = 200
    noise_multiplier = 1.0
    delta = 1.6e-5

    server_pipe_endpoints = {}

    dataset = 'MNIST'
    print(f'Using dataset: {dataset}')
    train_batch_size = 128
    test_batch_size = 128

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
        transform=transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,))
                ])
        for _, client in clients.items():
            executor.submit(initialize_client(client, dataset, train_batch_size, test_batch_size, transform))
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

        for _, client in clients.items():
            client.front_model.to(client.device)
            client.back_model.to(client.device)
            client.iterator = iter(client.train_DataLoader)


        for _, client in clients.items():
            # attaching PrivacyEngine to front model
            client.front_privacy_engine = PrivacyEngine()
            # client.front_model = ModuleValidator.fix(client.front_model)
            client.front_optimizer = optim.Adadelta(client.front_model.parameters(), lr=0.1)
            client.front_model, client.front_optimizer, client.train_DataLoader = \
                client.front_privacy_engine.make_private(
                module=client.front_model,
                data_loader=client.train_DataLoader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0,
                optimizer=client.front_optimizer,
            )


            # Attaching PrivacyEngine to back model
            # initialize privacy accountant
            client.back_accountant = RDPAccountant()

            # wrap model
            # client.back_model = ModuleValidator.fix(client.back_model)
            client.back_model = GradSampleModule(client.back_model)

            # initialize back optimizer
            client.back_optimizer = optim.Adadelta(client.back_model.parameters(), lr=0.1)

            # wrap optimizer
            client.back_optimizer = DPOptimizer(
                optimizer=client.back_optimizer,
                noise_multiplier=noise_multiplier, # same as make_private arguments
                max_grad_norm=1.0, # same as make_private arguments
                expected_batch_size=train_batch_size # if you're averaging your gradients, you need to know the denominator
            )

            # attach accountant to track privacy for an optimizer
            client.back_optimizer.attach_step_hook(
                client.back_accountant.get_optimizer_hook_fn(
                # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
                sample_rate=train_batch_size/60000
                )
            )


        # Training
        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch+1}:')


            start = time.time()
            # call forward prop for each client         
            for _, client in clients.items():
                executor.submit(client.forward_front())
            end = time.time()
            time_taken['forward_front'] += end-start


            start = time.time()
            # send activations to the server
            for _, client in clients.items():
                executor.submit(client.send_remote_activations1())
            end = time.time()
            time_taken['send_remote_activations1'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.get_remote_activations2())
            end = time.time()
            time_taken['get_remote_activations2'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.forward_back())
            end = time.time()
            time_taken['forward_back'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.calculate_loss())
            end = time.time()
            time_taken['calculate_loss'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.calculate_train_acc())
            end = time.time()
            time_taken['calculate_train_acc'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.backward_back())
            end = time.time()
            time_taken['backward_back'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.send_remote_activations2_grads())
            end = time.time()
            time_taken['send_remote_activations2_grads'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.get_remote_activations1_grads())
            end = time.time()
            time_taken['get_remote_activations1_grads'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.backward_front())
            end = time.time()
            time_taken['backward_front'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.step())
            end = time.time()
            time_taken['step'] += end-start


            start = time.time()
            for _, client in clients.items():
                executor.submit(client.zero_grad())
            end = time.time()
            time_taken['zero_grad'] += end-start


            train_acc = 0
            for _, client in clients.items():
                train_acc += client.train_acc[-1]
            train_acc = train_acc/num_clients
            overall_acc.append(train_acc)

            
            for _, client in clients.items():
                front_epsilon, front_best_alpha = client.front_privacy_engine.accountant.get_privacy_spent(delta=delta)
                back_epsilon, back_best_alpha = client.back_accountant.get_privacy_spent(delta=delta)
                print(f"([{client.id}] ε = {front_epsilon:.2f}, δ = {delta}) for α = {front_best_alpha}")
                print(f"([{client.id}] ε = {back_epsilon:.2f}, δ = {delta}) for α = {back_best_alpha}")


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


    print('\n')
    for func in time_taken:
        print(f'{func}: {(time_taken[func]/num_epochs):3f}')


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
