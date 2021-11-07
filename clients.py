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


# unused
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


# unused
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    num_clients = 1
    num_epochs = 10

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
            client.front_optimizer = optim.Adadelta(client.front_model.parameters(), lr=0.1)
            client.back_optimizer = optim.Adadelta(client.back_model.parameters(), lr=0.1)
            client.iterator = iter(client.train_DataLoader)


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
                executor.submit(client.zero_grad())
            end = time.time()
            time_taken['zero_grad'] += end-start

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


            train_acc = 0
            for _, client in clients.items():
                train_acc += client.train_acc[-1]
            train_acc = train_acc/num_clients
            overall_acc.append(train_acc)


        # Testing
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
            executor.submit(client.calculate_test_acc())


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
