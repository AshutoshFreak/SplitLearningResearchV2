import os
import random
import string
import socket
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
import argparse
import copy



SEED = 2647
random.seed(SEED)
torch.manual_seed(SEED)


# sets client attributes passed to the function
def initialize_client(client, dataset, batch_size, test_batch_size, tranform=None):
    client.load_data(args.dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Split Learning Research Simulation entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--number-of-clients",
        type=int,
        default=10,
        metavar="C",
        help="Number of Clients",
    )
    parser.add_argument(
        "--server-side-tuning",
        # action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        metavar="SST",
        help="State if server side tuning needs to be done",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "--server-sigma",
        type=float,
        default=0,
        metavar="SS",
        help="Noise multiplier for central layers",
    )
    parser.add_argument(
        "-g",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="G",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(        # needs to be implemented
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="States dataset to be used",
    )
    args = parser.parse_args()
    return args


# main
if __name__ == "__main__":

    
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_arguments()
    results = []

    # pipe endpoints for process communication through common memory space meant for server
    server_pipe_endpoints = {}

    # Choose dataset. 
    # See list of currently available datasets in utils/datasets.py
    print(f'Using dataset: {args.dataset}')

    # tracks average training accuracy and loss of clients per epoch
    overall_acc = []
    overall_loss = []

    # Generate random clients. returns dict with client id as key
    # and Client object as value. Initialization of clients is done later
    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')


    # split dataset between clients
    print('Splitting dataset...', end='')
    split_dataset(args.dataset, client_ids)
    print('Done')

    print(f'Random client ids:{str(client_ids)}')

    # executor.submit will enable each function to run in separate threads for each client
    with Executor() as executor:
        # define normalization transform
        # transform=transforms.Compose([
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #         ])
        transform = None


        ## client object initialization phase
        print('Initializing clients...')
        # all clients concurrently create dataloaders
        for _, client in clients.items():
            executor.submit(initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
        # initialization phase complete
        print('Client Intialization complete.')


        # all clients connect to the server
        for _, client in clients.items():
            executor.submit(client.connect_server())

        for client_id in clients:
            server_pipe_endpoints[client_id] = clients[client_id].server_socket


        # start server as a child process and provide client pipe endpoints
        p = multiprocessing.Process(target=server.main, args=(server_pipe_endpoints, args))
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


        # [server side tuning]
        if args.server_side_tuning:
            # 1 client is kept as a dummy client and is stripped from 'clients' dict
            # dummy client is the first client generated by generate_random_clients
            # *edge cases not considered. args.number_of_clients should be > 1*
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
            client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)

        # [server side tuning]
        if args.server_side_tuning:
            # initialize front optimizer for dummy client
            dummy_client.front_optimizer = optim.SGD(dummy_client.front_model.parameters(), lr=args.lr, momentum=0.9)


        # [Differential Privacy]
        # Update front_model, front_optimizer and train_Dataloader to be Differentially Private
        for _, client in clients.items():
            client.front_model, client.front_optimizer, client.train_DataLoader = \
                client.front_privacy_engine.make_private(
                module=client.front_model,
                data_loader=client.train_DataLoader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
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
            client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=args.lr, momentum=0.9)

            # # [Differential Privacy]
            # # wrap back optimizer
            # client.back_optimizer = DPOptimizer(
            #     optimizer=client.back_optimizer,
            #     sigma=args.sigma, # same as in make_private arguments
            #     max_grad_norm=1.0, # same as in make_private arguments
            #     expected_batch_size=args.batch_size # if you're averaging your gradients, you need to know the denominator
            # )

            # # attach accountant to track privacy for back optimizer
            # client.back_optimizer.attach_step_hook(
            #     client.back_accountant.get_optimizer_hook_fn(
            #     # this is an important parameter for privacy accounting. Should be equal to batch_size / len(args.dataset)
            #     sample_rate=args.batch_size/len(client.train_DataLoader.dataset)
            #     )
            # )
        
        # [server side tuning]
        if args.server_side_tuning:
            # initialize back optimizer for dummy client
            dummy_client.back_optimizer = optim.SGD(dummy_client.back_model.parameters(), lr=args.lr, momentum=0.9)


        # calculate number of iterations
        # Assume each client has exactly same number of datapoints
        # take number of datapoints of first client and divide with batch_size
        first_client = clients[client_ids[0]]
        num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)
        num_test_iterations = ceil(len(first_client.test_DataLoader.dataset)/args.batch_size)

        # Communicate epochs and number of iterations to server before training
        send_object(first_client.socket, (num_iterations, num_test_iterations))


        # Training
        for epoch in range(args.epochs):
            # # initialize optimizer for each client
            # for _, client in clients.items():
            #     client.front_model = ModuleValidator.fix(client.front_model)
            #     client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)
            # set iterator for each client and set running loss to 0
            for _, client in clients.items():
                client.iterator = iter(client.train_DataLoader)
                client.running_loss = 0

            for iteration in range(num_iterations):
                print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
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
                front_epsilon, front_best_alpha = client.front_privacy_engine.accountant.get_privacy_spent(delta=args.delta)
                client.front_epsilons.append(front_epsilon)
                client.front_best_alphas.append(front_best_alpha)
                print(f"([{client.id}] ε = {front_epsilon:.2f}, δ = {args.delta}) for α = {front_best_alpha}")


            # [server side tuning]
            if args.server_side_tuning:
                dummy_client.iterator = iter(dummy_client.train_DataLoader)
                dummy_client.running_loss = 0

                for iteration in range(num_iterations):
                    print(f'\r[Server side tuning] Epoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
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
                # train_acc = train_acc/args.number_of_clients
                # overall_acc.append(train_acc)


            # Testing
            with torch.no_grad():
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


                    # for _, client in clients.items():
                    #     executor.submit(client.calculate_loss())


                    for _, client in clients.items():
                        client.test_acc[-1] += client.calculate_test_acc()
                
                for _, client in clients.items():
                    client.test_acc[-1] /= num_test_iterations
                    overall_acc[-1] += client.test_acc[-1]
                
                overall_acc[-1] /= len(clients)
                # print(f'Acc for epoch {epoch+1}: {overall_acc[-1]}')
                print(f'Test Acc: {overall_acc[-1]}')


    for client_id, client in clients.items():
        plt.plot(list(range(args.epochs)), client.test_acc, label=f'{client_id} (Max:{max(client.test_acc):.4f})')
    plt.plot(list(range(args.epochs)), overall_acc, label=f'Average (Max:{max(overall_acc):.4f})')
    plt.title(f'{args.number_of_clients} Clients: Test Accuracy vs. Epochs')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ioff()
    plt.savefig(f'./results/test_acc_vs_epoch/{args.number_of_clients}_clients_{args.epochs}_epochs_{args.batch_size}_batch.png', bbox_inches='tight')
    plt.show()


    plt.plot(list(range(args.epochs)), first_client.front_epsilons)
    plt.title(f'{args.number_of_clients} Clients: Epsilon vs. Epochs')
    plt.ylabel('Epsilon')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ioff()
    plt.savefig(f'./results/epsilon_vs_epoch/{args.number_of_clients}_clients_{args.epochs}_epochs_{args.batch_size}_batch.png', bbox_inches='tight')
    plt.show()


    # picking up a random client and testing it's accuracy on overall test dataset
    random_clients_overall_acc = {}
    for random_client_id in clients:
        # random_client_id = random.choice(client_ids)
        # communicate the random client to the server
        send_object(first_client.socket, random_client_id)
        # ignoring creating a deepcopy of random client due to TypeError: cannot pickle '_thread.lock' object,
        # and I don't have time to fix it right now. Must be fixed for a clean and less buggy code in future
        # random_client = copy.deepcopy(clients[random_client_id])
        random_client = clients[random_client_id]
        random_client_overall_acc = 0
        random_client.test_acc = []


        with torch.no_grad():
            for _, client in clients.items():
                random_client.test_DataLoader = client.test_DataLoader
                random_client.iterator = iter(random_client.test_DataLoader)
                num_test_iterations = ceil(len(random_client.test_DataLoader.dataset)/args.batch_size)
                send_object(first_client.socket, num_test_iterations)
                random_client.test_acc.append(0)
                for iteration in range(num_test_iterations):
                    print(f'\rClient: {client.id}, Iteration: {iteration+1}/{num_test_iterations}', end='')

                    # forward prop for front model at random client
                    random_client.forward_front()

                    # send activations to the server at random client
                    random_client.send_remote_activations1()

                    # get remote activations from server at random client
                    random_client.get_remote_activations2()

                    # forward prop for back model at random client
                    random_client.forward_back()

                    # calculate test accuracy for random client
                    random_client.test_acc[-1] += random_client.calculate_test_acc()

                random_client.test_acc[-1] /= num_test_iterations
                random_client_overall_acc += random_client.test_acc[-1]

            random_client_overall_acc /= len(clients)
            random_clients_overall_acc[random_client_id] = random_client_overall_acc
            # print(f'Acc for epoch {epoch+1}: {overall_acc[-1]}')

    # print(f'Test acc for random clients: {random_clients_overall_acc}')
    for client_id in random_clients_overall_acc:
        print(f'{client_id}: {random_clients_overall_acc[client_id]}')

    # for client_id, client in clients.items():
    #     plt.plot(list(range(args.epochs)), client.front_epsilons)
    # plt.plot(list(range(args.epochs)), overall_loss)
    # plt.title(f'{args.number_of_clients} Clients: Train Loss vs. Epochs')
    # plt.ylabel('Train Loss')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.savefig(f'./results/train_loss_vs_epoch/{args.number_of_clients}_clients_{args.epochs}_epochs_{args.batch_size}_batch.png', bbox_inches='tight')
    # plt.show()


    # print('Test accuracy for each client:')
    # for client_id, client in clients.items():
    #         print(f'{client_id}:{client.test_acc}')
