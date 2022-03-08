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
from math import ceil
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor as Executor
from utils.split_dataset import split_dataset
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
import matplotlib.pyplot as plt
import time
import server
import multiprocessing
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
import torch.optim as optim


SEED = 2647
random.seed(SEED)
torch.manual_seed(SEED)


# sets client attributes passed to the function
def initialize_client(client, dataset, train_batch_size, test_batch_size, tranform=None):
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(train_batch_size, test_batch_size)


# main
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # num_clients should be > 1 for server side tuning.
    # need to put 'if server_side_tuning == True' feature
    num_clients = 10
    num_epochs = 1

    # [Differential Privacy]
    noise_multiplier = 1
    delta = 1e-5

    # pipe endpoints for process communication through common RAM space
    server_pipe_endpoints = {}

    # Choose dataset. 
    # See list of currently available datasets in utils/datasets.py
    dataset = 'MNIST'
    print(f'Using dataset: {dataset}')

    # define train and test batch size
    train_batch_size = 32
    test_batch_size = 32

    # tracks average training accuracy and loss of clients per epoch
    overall_acc = []
    overall_loss = []

    # Generate random clients. returns dict with client id as key
    # and Client object as value. Initialization of clients is done later
    print('Generating random clients...', end='')
    clients = generate_random_clients(num_clients)
    client_ids = list(clients.keys())    
    print('Done')


    # split dataset between clients
    print('Splitting dataset...', end='')
    split_dataset(dataset, client_ids)
    print('Done')

    print(f'Random client ids:{str(client_ids)}')

    # executor.submit will enable each function to run in separate threads for each client
    with Executor() as executor:
        # define normalization transform
        transform=transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,))
                ])


        ## client object initialization phase
        print('Initializing clients...')
        # all clients concurrently create dataloaders
        for _, client in clients.items():
            executor.submit(initialize_client(client, dataset, train_batch_size, test_batch_size, transform))
        # initialization phase complete
        print('Client Intialization complete.')


        # all clients connect to the server
        for _, client in clients.items():
            executor.submit(client.connect_server())

        for client_id in clients:
            server_pipe_endpoints[client_id] = clients[client_id].server_socket


        # start server as a child process and provide client pipe endpoints
        p = multiprocessing.Process(target=server.main, args=(server_pipe_endpoints,))
        p.start()
        

        # all clients get model from the server
        print('Getting model from server...', end='')
        for _, client in clients.items():
            executor.submit(client.get_model())
        print('Done')


        # # for [manual] verification of successful transfer, printing front model
        # for _, client in clients.items():
        #     print(client.front_model)

        for _, client in clients.items():
            client.front_model.to(client.device)
            client.back_model.to(client.device)

        # [Server side tuning]
        # 1 client is kept as a dummy client and is stripped from 'clients' dict
        # dummy client is the first client generated by generate_random_clients
        # *edge cases not considered. num_clients should be > 1*
        dummy_client_id = client_ids[0]
        client_ids = client_ids[1:]
        dummy_client = clients[dummy_client_id]
        clients.pop(dummy_client_id)


        # [Differential Privacy]
        # Initialize front PrivacyEngine
        for _, client in clients.items():
            client.front_privacy_engine = PrivacyEngine()


        # initialize optimizer for each client
        for _, client in clients.items():
            client.front_model = ModuleValidator.fix(client.front_model)
            client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=0.05)

        
        # [server side tuning]
        # initialize front optimizer for dummy client
        dummy_client.front_optimizer = optim.SGD(dummy_client.front_model.parameters(), lr=0.05)


        # [Differential Privacy]
        # Update front_model, front_optimizer and train_Dataloader to be Differentially Private
        for _, client in clients.items():
            client.front_model, client.front_optimizer, client.train_DataLoader = \
                client.front_privacy_engine.make_private(
                module=client.front_model,
                data_loader=client.train_DataLoader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0,
                optimizer=client.front_optimizer,
            )

        # # [Differential Privacy]
        for _, client in clients.items():
            
            # # Attaching PrivacyEngine to back model
            # # initialize privacy accountant
            # client.back_accountant = RDPAccountant()

            # # [Differential Privacy]
            # # wrap model
            # client.back_model = GradSampleModule(client.back_model)

            # initialize back optimizer
            client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=0.05)

            # # [Differential Privacy]
            # # wrap back optimizer
            # client.back_optimizer = DPOptimizer(
            #     optimizer=client.back_optimizer,
            #     noise_multiplier=noise_multiplier, # same as in make_private arguments
            #     max_grad_norm=1.0, # same as in make_private arguments
            #     expected_batch_size=train_batch_size # if you're averaging your gradients, you need to know the denominator
            # )

            # # attach accountant to track privacy for back optimizer
            # client.back_optimizer.attach_step_hook(
            #     client.back_accountant.get_optimizer_hook_fn(
            #     # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
            #     sample_rate=train_batch_size/len(client.train_DataLoader.dataset)
            #     )
            # )
        
        # [server side tuning]
        # initialize back optimizer for dummy client
        dummy_client.back_optimizer = optim.SGD(dummy_client.back_model.parameters(), lr=0.05)


        # calculate number of iterations
        # Assume each client has exactly same number of datapoints
        # take number of datapoints of first client and divide with batch_size
        first_client = clients[client_ids[0]]
        num_iterations = ceil(len(first_client.train_DataLoader.dataset)/train_batch_size)
        num_test_iterations = ceil(len(first_client.test_DataLoader.dataset)/train_batch_size)

        # Communicate epochs and number of iterations to server before training
        send_object(first_client.socket, (num_epochs, num_iterations, num_test_iterations))


        # Training
        for epoch in range(num_epochs):
            # # initialize optimizer for each client
            # for _, client in clients.items():
            #     client.front_model = ModuleValidator.fix(client.front_model)
            #     client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=0.05)
            # set iterator for each client and set running loss to 0
            for _, client in clients.items():
                client.iterator = iter(client.train_DataLoader)
                client.running_loss = 0

            for iteration in range(num_iterations):
                print(f'\nEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}')
                # forward prop for front model at each client
                for _, client in clients.items():
                    executor.submit(client.forward_front())


                # send activations to the server at each client
                for _, client in clients.items():
                    executor.submit(client.send_remote_activations1())


                # get remote activations from server at each client
                for _, client in clients.items():
                    executor.submit(client.get_remote_activations2())


                # forward prop for back model at each client
                for _, client in clients.items():
                    executor.submit(client.forward_back())


                # calculate loss at each client
                for _, client in clients.items():
                    executor.submit(client.calculate_loss())


                # # calculate training accuracy at each client
                # for _, client in clients.items():
                #     executor.submit(client.calculate_train_acc())


                # backprop for back model at each client
                for _, client in clients.items():
                    executor.submit(client.backward_back())


                # send gradients to server
                for _, client in clients.items():
                    executor.submit(client.send_remote_activations2_grads())


                # get gradients from server
                for _, client in clients.items():
                    executor.submit(client.get_remote_activations1_grads())


                # backprop for front model at each client
                for _, client in clients.items():
                    executor.submit(client.backward_front())


                # update weights of both front and back model at each client
                for _, client in clients.items():
                    executor.submit(client.step())


                # zero out all gradients at each client
                for _, client in clients.items():
                    executor.submit(client.zero_grad())


                # add losses for each iteration
                for _, client in clients.items():
                    client.running_loss += client.loss


                # [Differential Privacy] get back epsilon with delta values
                for _, client in clients.items():
                    front_epsilon, front_best_alpha = client.front_privacy_engine.accountant.get_privacy_spent(delta=delta)
                    # back_epsilon, back_best_alpha = client.back_accountant.get_privacy_spent(delta=delta)
                    print(f"([{client.id}] ε = {front_epsilon:.2f}, δ = {delta}) for α = {front_best_alpha}")
                    # print(f"([{client.id}] ε = {back_epsilon:.2f}, δ = {delta}) for α = {back_best_alpha}")
            

            # [server side tuning]
            print('server side tuning')
            dummy_client.iterator = iter(dummy_client.train_DataLoader)
            dummy_client.running_loss = 0

            for iteration in range(num_iterations):
                print(f'\nEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}')
                # forward prop for front model at dummy client
                dummy_client.forward_front()

                # send activations to the server at dummy client
                dummy_client.send_remote_activations1()

                # get remote activations from server at dummy client
                dummy_client.get_remote_activations2()

                # forward prop for back model at dummy client
                dummy_client.forward_back()

                # calculate loss at dummy client
                dummy_client.calculate_loss()

                # backprop for back model at dummy client
                dummy_client.backward_back()

                # send gradients to server
                dummy_client.send_remote_activations2_grads()

                # get gradients from server
                dummy_client.get_remote_activations1_grads()

                # backprop for front model at dummy client
                dummy_client.backward_front()

                # update weights of both front and back model at dummy client
                dummy_client.step()

                # zero out all gradients at dummy client
                dummy_client.zero_grad()

                # add losses for each iteration
                dummy_client.running_loss += dummy_client.loss

            overall_loss.append(0)
            avg_loss = 0
            # average out losses over all iterations for a single loss per epoch
            for _, client in clients.items():
                loss = client.running_loss/num_iterations
                client.losses.append(loss)
                overall_loss[-1] += loss
            overall_loss[-1] /= num_iterations

            

                # train_acc = 0
                # # average out accuracy of all clients
                # for _, client in clients.items():
                #     train_acc += client.train_acc[-1]
                # train_acc = train_acc/num_clients
                # overall_acc.append(train_acc)


        # Testing
        test_acc = 0
        overall_acc.append(0)
        for _, client in clients.items():
            client.test_acc.append(0)
        for iteration in range(num_test_iterations):
            # Setting up iterator for testing
            for _, client in clients.items():
                client.iterator = iter(client.test_DataLoader)


            # call forward prop for each client
            for _, client in clients.items():
                executor.submit(client.forward_front())


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
                client.test_acc[-1] += client.calculate_test_acc()
            
        for _, client in clients.items():
            client.test_acc[-1] /= num_test_iterations
            overall_acc[-1] += client.test_acc[-1]
        
        overall_acc[-1] /= num_clients
        # print(f'Acc for epoch {epoch+1}: {overall_acc[-1]}')
        print(f'Test Acc: {overall_acc[-1]}')


    # for client_id, client in clients.items():
    #     plt.plot(list(range(num_epochs)), client.test_acc, label=f'{client_id} (Max:{max(client.test_acc):.4f})')
    # plt.plot(list(range(num_epochs)), overall_acc, label=f'Average (Max:{max(overall_acc):.4f})')
    # plt.title(f'{num_clients} Clients: Test Accuracy vs. Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()
    # # plt.savefig(f'./results/train_acc_vs_epoch/{num_clients}_clients_{num_epochs}_epochs.png', bbox_inches='tight')
    # plt.show()

    # for client_id, client in clients.items():
    #     plt.plot(list(range(num_epochs)), client.losses)
    # plt.plot(list(range(num_epochs)), overall_loss)
    # plt.title(f'{num_clients} Clients: Train Loss vs. Epochs')
    # plt.ylabel('Train Loss')
    # plt.xlabel('Epochs')
    # plt.legend()
    # # plt.savefig(f'./results/train_acc_vs_epoch/{num_clients}_clients_{num_epochs}_epochs.png', bbox_inches='tight')
    # plt.show()


    # print('Test accuracy for each client:')
    # for client_id, client in clients.items():
    #         print(f'{client_id}:{client.test_acc}')
