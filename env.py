from torch.random import seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
import torch
import random

from typing import List, Dict, Tuple
import numpy as np

from nodes import *
from fedProtocol import FedProtocol
from preprocess import DatasetManager

from tqdm import tqdm
from ray import tune
import os

class FederatedLearningEnvironment(): 
    '''
    Very Basic Federated Learning environment. There is a server and client and they are connected in
    hub-and-spoke topology. Every round, only subset of client is participating. This is a static environment with flexible communication channel
    '''
    def __init__(self, n_clients: int, communication_round: int, seed: int, train_ratio: float, dataset_dir: str = "dataset", device = 'cpu') -> None:
        '''
        params:
        - n_clients: How many clients in the system
        - communication_round: How long is the training
        - seed: for numpy and torch
        - train_ratio: Proportion of training data with respect to total data
        - C: Fraction of participating client
        '''
        self.n_clients = n_clients
        self.communication_round = communication_round
        self.seed = seed
        self.train_ratio = train_ratio
        self.device = device
        self.dataset_dir = dataset_dir

    def construct_nodes(self, train_loader: List[DataLoader], model_fn_client: Callable[[], torch.nn.Module], model_fn_server: Callable[[], torch.nn.Module], loss_fn: Any, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
    fedProtocol: FedProtocol, local_epoch: int, device = 'cpu', eval_loader: DataLoader = None,
    lr_scheduler_fn: Callable[[torch.optim.Optimizer], _LRScheduler] = lambda _ : None, participating_ratio: float = 1.0, eval_metrics="normal", **kwargs_client) -> Tuple[List[Client], Server]:
        '''
        params:
        - data from get_dataset and params configure the data distribution
        '''
        if eval_metrics == "legal":
            clients = [LegalClient(id, model_fn_client, fedProtocol, train_loader[id], loss_fn=loss_fn, optimizer_fn=optimizer_fn, device=device,
            local_epoch=local_epoch, eval_loader=eval_loader[id], lr_scheduler_fn=lr_scheduler_fn, **kwargs_client) for id in range(self.n_clients)]
        else:
            clients = [Client(id, model_fn_client, fedProtocol, train_loader[id], loss_fn=loss_fn, optimizer_fn=optimizer_fn, device=device,
            local_epoch=local_epoch, eval_loader=eval_loader[id], lr_scheduler_fn=lr_scheduler_fn) for id in range(self.n_clients)]
        server = Server(model_fn_server, fedProtocol, device=device, loss_fn=loss_fn)
        if np.isclose(participating_ratio, 1.0):
            return clients, server
        else:
            return (clients[:int(participating_ratio*self.n_clients)], clients[int(participating_ratio*self.n_clients):]), server

    def prepare_dataset(self, dataset_names: List[str], train_transforms: Dict[str, transforms.Compose], test_transforms: Dict[str, transforms.Compose], 
    options: Dict[str, Any], verbose:bool, batch_size: int, dataset_config: Dict[str, Any], batch_config: Dict[str, Any], use_ray: bool, frac_each_client: float)-> Tuple[List[DataLoader], DataLoader, DataLoader]:
        dm = DatasetManager(self.n_clients, train_transforms, test_transforms, options, verbose, dataset_config, self.train_ratio, use_ray, frac_each_client)
        client_data, eval_data = dm.load_and_distribute_dataset(dataset_names, self.dataset_dir)
        client_loader, eval_loader = dm.get_dataloader(client_data, batch_size, batch_config), dm.get_dataloader(eval_data, batch_size, batch_config)
        return client_loader, eval_loader

    def set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def start_training(self, dataset_names: List[str], train_transforms: Dict[str, transforms.Compose], test_transforms: Dict[str, transforms.Compose],
    dataset_options: Dict[str, Any], verbose:bool, batch_size: int, dataset_config: Any, batch_config: Dict[str, Any], model_fn_client: Callable[[], torch.nn.Module], model_fn_server: Callable[[], torch.nn.Module],
    loss_fn: Any, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer], fedProtocol: FedProtocol, local_epoch: int, checkpoint_dir=None, lr_scheduler_fn: Callable[[torch.optim.Optimizer], _LRScheduler] = lambda _ : None,
    use_ray: bool = True, participating_ratio: float = 1.0, eval_metrics="normal", frac_each_client=1.0, eval_method="one_client", **kwargs_client):
        '''
        FL Training upto a certain communication round
        params:
        - dataset_names: List of dataset that will be used for training (Can be more than one source)
        - train_transforms: transformation for the feature on the train dataset (Dict with dataset name as key)
        - test_transforms: transformation for the feature on the validation dataset and train dataset
        - dataset_options: Options for how the data should be distributed to the client side
        - verbose: Some handful statistics about the dataset
        - batch_size: self-explanatory
        - dataset_config: Additional parameters for data distribution such as dirichlet alpha
        - batch_config: configuration on the dataloader. May take lambda for the sampling for example
        - model_fn_client: Lambda for NN model for client side
        - model_fn_server: Lambda for NN model for server side
        - loss_fn: The object of the loss function
        - optimizer_fn: Lambda for optimizer
        - fedProtocol
        - local_epoch
        - participating_ratio: Proportion of client participating in the training. The unparticipating is a part of the FL model evaluation
        - eval_metrics (str): normal, legal
        '''

        self.set_seed()

        train_loader, eval_loader = self.prepare_dataset(dataset_names=dataset_names, train_transforms=train_transforms, test_transforms=test_transforms, 
        options=dataset_options, verbose=verbose, batch_size=batch_size, dataset_config=dataset_config, batch_config=batch_config, use_ray=use_ray, frac_each_client=frac_each_client)

        clients, server = self.construct_nodes(train_loader=train_loader, model_fn_client=model_fn_client, model_fn_server=model_fn_server, loss_fn=loss_fn, optimizer_fn=optimizer_fn, fedProtocol=fedProtocol, 
        local_epoch=local_epoch, device=self.device, eval_loader=eval_loader, lr_scheduler_fn=lr_scheduler_fn, participating_ratio=participating_ratio, eval_metrics=eval_metrics, **kwargs_client)
        init_round = 0

        if isinstance(clients, tuple):
            participating_clients, unparticipating_clients = clients
        else:
            participating_clients = clients
            unparticipating_clients = None
        
        if checkpoint_dir: #If specified, load
            server.model = server.model.to('cpu')
            server.model.load_state_dict(torch.load(checkpoint_dir))
            server.model = server.model.to(server.device)
        if use_ray:
            R = range(init_round, self.communication_round + 1)
        else:
            R = tqdm(range(init_round, self.communication_round + 1))
        for c_round in R:
            server.broadcast_weight(participating_clients, is_unpart=False)
            #Eval for previous round
            if c_round != init_round:
                if eval_metrics == "legal":
                    if eval_method == "all_client":
                        eval_loss, eval_acc, eval_DP, eval_EO = tuple(map(np.mean, zip(*[client.eval(loader="train") for client in selected_clients])))
                    else:
                        eval_loss, eval_acc, eval_DP, eval_EO = tuple(map(np.mean, zip(*[client.eval() for client in selected_clients])))
                else:
                    eval_loss, eval_acc= tuple(map(np.mean, zip(*[client.eval() for client in selected_clients])))
                if unparticipating_clients is not None:
                    server.broadcast_weight(unparticipating_clients, is_unpart=True)
                    if eval_metrics == "legal":
                        eval_loss_unparticipating, eval_acc_unparticipating, eval_DP_unparticipating, eval_EO_unparticipating = tuple(map(np.mean, zip(*[client.eval(loader="all") for client in unparticipating_clients])))
                    else:
                        eval_loss_unparticipating, eval_acc_unparticipating = tuple(map(np.mean, zip(*[client.eval(loader="all") for client in unparticipating_clients])))
                else:
                    eval_loss_unparticipating, eval_acc_unparticipating, eval_DP_unparticipating, eval_EO_unparticipating = 0.0, 0.0, 0.0, 0.0
                if c_round == self.communication_round:
                    try:
                        if not os.path.exists("model_checkpoint"):
                            os.makedirs("model_checkpoint")
                        torch.save(server.model.state_dict(), f"model_checkpoint/global_model round = {c_round}.pt")
                    except:
                        print("Save file unsuccessful")
            
            selected_clients = server.select_clients(participating_clients)
            for client in selected_clients:
                client.local_compute()
            server_info = server.aggregate_weights(selected_clients, global_state={'c_round': c_round})

            if c_round != init_round:
                if use_ray:
                    if eval_metrics == "legal":
                        if server_info and "lam" in server_info:
                            dual_var = server_info["lam"]
                            tune.report(eval_loss=eval_loss, eval_acc=eval_acc, eval_loss_unpart=eval_loss_unparticipating, eval_acc_unpart=eval_acc_unparticipating, eval_DP = eval_DP, eval_EO = eval_EO, eval_DP_unparticipating = eval_DP_unparticipating, eval_EO_unparticipating= eval_EO_unparticipating, dual_var=dual_var)
                        else:
                            tune.report(eval_loss=eval_loss, eval_acc=eval_acc, eval_loss_unpart=eval_loss_unparticipating, eval_acc_unpart=eval_acc_unparticipating, eval_DP = eval_DP, eval_EO = eval_EO, eval_DP_unparticipating = eval_DP_unparticipating, eval_EO_unparticipating= eval_EO_unparticipating)
                    else:
                        tune.report(eval_loss=eval_loss, eval_acc=eval_acc, eval_loss_unpart=eval_loss_unparticipating, eval_acc_unpart=eval_acc_unparticipating)
                else:
                    R.set_description(f"eval_acc = {eval_acc}, eval_DP = {eval_DP}, eval_EO = {eval_EO}")
            
        