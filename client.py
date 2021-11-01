import os
import torch
import torch.nn.functional as F
import multiprocessing
from threading import Thread
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
from utils.split_dataset import DatasetFromSubset
import pickle
import queue
import struct
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

# class SeverProtocol:


class Client(Thread):
    # def __init__(self, id, loop_time=1/60, *args, **kwargs):
    def __init__(self, id, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        # self.q = queue.Queue()
        # self.timeout = loop_time
        # Thread.__init__(self)
        self.id = id
        self.front_model = []
        self.back_model = []
        self.losses = []
        self.train_dataset = None
        self.test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.iterator = None
        self.activations1 = None
        self.remote_activations1 = None
        self.output = None
        self.loss = None
        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.train_acc = []
        self.test_acc = []
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        

    def forward_front(self):
        self.data, self.targets = next(self.iterator)
        # self.data = self.data.reshape(-1, 784).to(self.device)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)

        self.activations1 = self.front_model(self.data)
        self.remote_activations1 = self.activations1.detach().requires_grad_(True)



    def forward_back(self):
        self.outputs = self.back_model(self.remote_activations2)


    def backward_front(self):
        self.activations1.backward(self.remote_activations1.grad)


    def backward_back(self):
        self.loss.backward()


    def calculate_loss(self):
        self.criterion = F.nll_loss
        self.loss = self.criterion(self.outputs, self.targets)
        self.losses.append(self.loss)


    def calculate_train_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            self.train_acc.append(100.0 * self.n_correct/self.n_samples)
            print(f'Acc: {self.train_acc[-1]}')


    def calculate_test_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            self.test_acc.append(100.0 * self.n_correct/self.n_samples)
            print(f'Acc: {self.test_acc[-1]}')


    def zero_grad(self):
        self.front_optimizer.zero_grad()
        self.back_optimizer.zero_grad()


    def step(self):
        self.front_optimizer.step()
        self.back_optimizer.step()

    # def forward_front():
    #     pass
    

    # def forward_back():
    #     pass

    
    # def backward_front():
    #     pass


    # def backward_back():
    #     pass


    # def onThread(self, function, *args, **kwargs):
    #     self.q.put((function, args, kwargs))


    # def run(self, *args, **kwargs):
    #     super(Client, self).run(*args, **kwargs)
    #     while True:
    #         try:
    #             function, args, kwargs = self.q.get(timeout=self.timeout)
    #             function(*args, **kwargs)
    #         except queue.Empty:
    #             self.idle()


    def idle(self):
        pass


    def load_data(self, dataset, transform):
        try:
            dataset_path = os.path.join(f'data/{dataset}/{self.id}')
        except:
            raise Exception(f'Dataset not found for client {self.id}')
        self.train_dataset = torch.load(f'{dataset_path}/train/{self.id}.pt')
        self.test_dataset = torch.load(f'{dataset_path}/test/{self.id}.pt')

        self.train_dataset = DatasetFromSubset(
            self.train_dataset, transform=transform
        )
        self.test_dataset = DatasetFromSubset(
            self.test_dataset, transform=transform
        )


    def create_DataLoader(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_DataLoader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.train_batch_size,
                                                shuffle=True)
        self.test_DataLoader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.test_batch_size,
                                                shuffle=True)


    def connect_server(self, host='localhost', port=8000, BUFFER_SIZE=4096):
        # self.socket = socket.socket()
        self.socket, self.server_socket = multiprocessing.Pipe()
        print(f"[*] Client {self.id} connecting to {host}")
        # print(f"[*] Client {self.id} connecting to {host}:{port}")
        # try:
        #     self.socket.connect((host, port))
        #     print(f'[*] Client {self.id} connected! {self.id} address: {self.socket.getsockname()[0]}:{self.socket.getsockname()[1]}\n')
        #     self.socket.sendall(str.encode(self.id))
        #     return True

        # except socket.error as e:
        #     print(f'[*] Client {self.id} failed to connect to the server.\n{e}')
        #     return False


    def disconnect_server(self) -> bool:
        if not is_socket_closed(self.socket):
            self.socket.close()
            return True
        else:
            return False


    def send_remote_activations1(self):
        send_object(self.socket, self.remote_activations1)
    

    def send_remote_activations2_grads(self):
        send_object(self.socket, self.remote_activations2.grad)


    def get_remote_activations1_grads(self):
        self.remote_activations1.grad = get_object(self.socket)


    def get_remote_activations2(self):
        self.remote_activations2 = get_object(self.socket)


    # def train_model(self):
    #     forward_front_model()
    #     send_activations_to_server()
    #     forward_back_model()
    #     loss_calculation()
    #     backward_back_model()
    #     send_gradients_to_server()
    #     backward_front_model()


    # def _getModel(self):
    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']
    
    
    # def getModel(self):
    #     self.onThread(self._getModel)    
