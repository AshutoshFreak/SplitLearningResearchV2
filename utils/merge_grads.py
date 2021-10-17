import torch
def merge_grads(params):
    # params = [params_client1,
    #           params_client2,
    #           params_client3
    #           ...
    #          ]
    num_clients = len(params)
    for col in zip(*params):
        avg = 0
        for param in col:
            avg += param.grad
        avg /= num_clients
        for param in col:
            param.grad = avg.clone() 
