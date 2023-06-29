import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler


from typing import Any, List, Dict
from abc import ABCMeta, abstractmethod
import numpy as np
import random
import copy

from Losses import weighted_loss

class FedProtocol(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def local_compute(self, client, loader: DataLoader):
        pass
    
    @abstractmethod
    def aggregation(self, server: Any, participating_clients: List[Any], global_state: Any):
        pass

    @abstractmethod
    def select_client(self, clients, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def broadcast_weight(self, clients, global_model): 
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())

    def set_server_model(self, model): #Called every broadcast weight
        self.global_model = model
    
    def post_processing(self, server, clients): #Called at the end of the training, default do nothing
        return 


class FedAvg(FedProtocol):
    def __init__(self, C, grad_clip = None, excluded_params: List[str] = [], included_params: List[str] = []) -> None:
        self.C = C
        self.grad_clip = grad_clip
        self.excluded_params = excluded_params #Server still aggregate all parameters but clients will not receive the excluded params unless they not participating
        self.included_params = included_params #The difference with excluded_params is that this will effect the local_compute as well. [] means local compute for all parameters
        #If both names present, it will not broadcasted to the participating clients while the parameters is being modified only locally. 
    
    def local_compute(self, client, loader: DataLoader, lr_schedule: _LRScheduler):
        model: Module = client.model
        local_epoch: int = client.local_epoch
        device = client.device
        optimizer: Optimizer = client.optimizer
        loss_fn = client.loss_fn

        old_model: Module = copy.deepcopy(model)
        client.dW = copy.deepcopy(model)

        model.train()
        if self.included_params:
            for k, v in model.named_parameters():
                if any([s in k for s in self.included_params]):
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        correct = 0
        train_loss = 0.0
        for e in range(local_epoch):
            for batches in loader:
                if len(batches) == 2:
                    X, y = batches
                else:
                    X, y, _ = batches
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                train_loss += loss.item()
                optimizer.step()
                pred = out.data.max(1)[1]
                correct += pred.eq(y.view(-1)).sum().item()
        if lr_schedule:
            lr_schedule.step()
        
        with torch.no_grad():
            for k,v in model.state_dict().items():
                client.dW.state_dict()[k].data.copy_(v.clone() - old_model.state_dict()[k].clone())
        
        return train_loss/(local_epoch*len(loader.dataset)), correct/(local_epoch*len(loader.dataset))

    def aggregation(self, server: Any, participating_clients: List[Any], global_state={}):
        global_model = server.model
        client_gradients = [client.dW for client in participating_clients]
        #client_models = [client.model for client in participating_clients]
        if "client_weights" not in global_state:
            client_weights = np.array([len(client.train_loader.dataset) for client in participating_clients], dtype=float)
            client_weights = client_weights/sum(client_weights)
        else:
            client_weights = global_state["client_weights"]
        self.client_weights = client_weights
        with torch.no_grad():
            for key in global_model.state_dict().keys():
                temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for client_weight, client_gradient in zip(client_weights, client_gradients):
                    temp += client_weight*client_gradient.state_dict()[key]
                global_model.state_dict()[key].data.copy_(global_model.state_dict()[key].data.clone() + temp)
    
    def select_client(self, clients):
        return random.sample(clients, int(self.C*len(clients)))
    
    def broadcast_weight(self, clients, global_model, **kwargs):
        with torch.no_grad():
            for key in global_model.state_dict().keys():
                if self.excluded_params and (not kwargs['is_unpart'] and any([s in key for s in self.excluded_params])):
                    # if not kwargs['is_unpart'] and ('bn' in key  or 'norm' in key or ('bias' not in key and 'weight' not in key)):
                    continue
                else:
                    for client in clients:
                        client.model.state_dict()[key].data.copy_(global_model.state_dict()[key])

    def __str__(self):
        return "FedAvg"

class FairFedAvgALM(FedAvg):
    def __init__(self, C, grad_clip, beta, b, eta, lam, sigma_dp=0.) -> None:
        '''
        g(w, lam) = lam(mu^s0_w - mu^s1_w) + beta/2(mu^s0_w - mu^s1_w)^2

        params:
        C: fraction of participating client
        grad_clip: gradient clipping value
        b: increase factor for beta and decrease factor for learning rate
        eta: learning rate
        sigma_dp: standard deviation of the Gaussian mechanism
        '''
        super().__init__(C, grad_clip)
        self.beta = beta
        self.b = b
        self.eta_init = eta
        self.eta = eta
        self.lam = lam
        self.sigma_dp = sigma_dp
    
    def primal_update(self, loss_all, lam, beta, loss_s0, loss_s1):
        return loss_all.mean() + lam*(loss_s0 - loss_s1) + beta/2. * (loss_s0 - loss_s1) * (loss_s0 - loss_s1)
    
    def dual_update(self, client, loss_s0, loss_s1, loss_all):
        client.lam =  client.lam + self.eta*(loss_s0 - loss_s1)

    def local_compute(self, client, loader: DataLoader, lr_schedule: _LRScheduler):
        model: Module = client.model
        local_epoch: int = client.local_epoch
        device = client.device
        optimizer: Optimizer = client.optimizer
        loss_fn = torch.nn.CrossEntropyLoss(reduce=False)

        old_model: Module = copy.deepcopy(model)
        client.dW = copy.deepcopy(model)
        model.train()

        correct = 0
        train_loss_val = 0.0

        for features, targets, protected in loader:
            features = features.to(device)
            targets = targets.to(device)
            protected = protected.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss_all = loss_fn(logits, targets)
            loss_s0 = loss_all[(protected==0)].mean()
            loss_s1 = loss_all[(protected==1)].mean()
            if loss_s0.isnan():
                loss_s0 = torch.zeros(1).float().to(device)
            if loss_s1.isnan():
                loss_s1 = torch.zeros(1).float().to(device)
            train_loss = loss_all.mean()
            train_loss_val += train_loss.item()

            pred = logits.data.max(1)[1]
            correct += pred.eq(targets.view(-1)).sum().item()
            # Primal Update
            loss = self.primal_update(loss_all, client.lam, self.beta, loss_s0, loss_s1)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

        #Dual update
        self.dual_update(client, loss_s0, loss_s1, loss_all)
        with torch.no_grad():
            for k,v in model.state_dict().items():
                client.dW.state_dict()[k].data.copy_(v.clone() - old_model.state_dict()[k].clone() + torch.randn(v.shape).to(device)*self.sigma_dp)
        if lr_schedule:
            lr_schedule.step()
        return train_loss_val/(local_epoch*len(loader.dataset)), correct/(local_epoch*len(loader.dataset))

    def aggregation(self, server: Any, participating_clients: List[Any], global_state={}):
        super().aggregation(server, participating_clients, global_state) #Primal aggregation
        self.lam_old = self.lam
        with torch.no_grad():
            for i, client in enumerate(participating_clients):
                self.lam += self.client_weights[i]*(client.lam - self.lam_old + self.sigma_dp*np.random.randn())
            self.beta = self.b*self.beta
            self.eta = self.eta/self.b
        return {'lam': self.lam}

    def broadcast_weight(self, clients, global_model, **kwargs):
        super().broadcast_weight(clients, global_model, **kwargs)
        for client in clients:
            client.lam = self.lam

class FedAvgFairALM(FairFedAvgALM):
    def __init__(self, C, grad_clip, b, eta, lam) -> None:
        super().__init__(C, grad_clip, 0, b, eta, lam)
    
    def primal_update(self, loss_all, lam, beta, loss_s0, loss_s1):
        return loss_all.mean() + lam*(loss_s0 - loss_s1) + self.eta*(loss_s0 + loss_s1)

class FairFed(FedAvg):
    def __init__(self, C, grad_clip, beta, fair_metric='DP') -> None:
        '''
        Implement https://arxiv.org/pdf/2110.00857.pdf
        Assuming full participation
        '''
        super().__init__(C, grad_clip)
        self.beta = beta
        self.fair_metric = fair_metric

    def local_compute(self, client, loader: DataLoader, lr_schedule: _LRScheduler):

        train_loss, train_acc, train_DP, train_EO = client.eval(loader="train")
        if self.fair_metric == "DP":
            client.local_DP = train_DP/100.
        else:
            client.local_DP = train_EO/100.
        
        model: Module = client.model
        local_epoch: int = client.local_epoch
        device = client.device
        optimizer: Optimizer = client.optimizer
        loss_fn = client.loss_fn

        model.train()
        old_model: Module = copy.deepcopy(model)
        client.dW = copy.deepcopy(model)

        correct = 0
        train_loss = 0.0
        for e in range(local_epoch):
            for X,y,_ in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                train_loss += loss.item()
                optimizer.step()
                pred = out.data.max(1)[1]
                correct += pred.eq(y.view(-1)).sum().item()
        if lr_schedule:
            lr_schedule.step()
        
        with torch.no_grad():
            for k,v in model.state_dict().items():
                client.dW.state_dict()[k].data.copy_(v.clone() - old_model.state_dict()[k].clone())
        
        return train_loss/(local_epoch*len(loader.dataset)), correct/(local_epoch*len(loader.dataset))

    def aggregation(self, server: Any, participating_clients: List[Any], global_state={}):
        with torch.no_grad():
            if not hasattr(self, "client_weights"):
                self.sample_weights = np.zeros(len(participating_clients), dtype=float)
                for client in participating_clients:
                    self.sample_weights[client.id] = len(client.train_loader.dataset)
                self.sample_weights = self.sample_weights/np.sum(self.sample_weights)
                self.client_weights = np.copy(self.sample_weights)

            global_DP = sum([self.sample_weights[client.id]*client.local_DP for client in participating_clients])
            deltas = np.zeros(len(participating_clients), dtype=float)
            for client in participating_clients:
                deltas[client.id] = abs(global_DP - client.local_DP)
            global_delta = np.sum(deltas)/len(participating_clients)
            for client in participating_clients:
                self.client_weights[client.id] = np.clip(self.client_weights[client.id] - self.beta*(deltas[client.id] - global_delta), 0, None)
            self.client_weights = self.client_weights/np.sum(self.client_weights)

            global_model = server.model
            client_gradients = [client.dW for client in participating_clients]
            for key in global_model.state_dict().keys():
                temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for client_weight, client_gradient in zip(self.client_weights, client_gradients):
                    temp += client_weight*client_gradient.state_dict()[key]
                global_model.state_dict()[key].data.copy_(global_model.state_dict()[key].data.clone() + temp) 
            
class FPFL(FairFedAvgALM):
    def __init__(self, C, grad_clip, beta, eta):
        super().__init__(C, grad_clip, beta, 1.00, eta, 0.)
        self.lam = torch.zeros(2)
    
    def primal_update(self, loss_all, lam, beta, loss_s0, loss_s1):
        return loss_all.mean() + lam[0]*torch.abs(loss_all.mean() - loss_s0) + lam[1]*torch.abs(loss_all.mean() - loss_s1) + beta*((torch.abs(loss_all.mean() - loss_s1))**2 + (torch.abs(loss_all.mean() - loss_s0))**2)/2.
    
    def dual_update(self, client, loss_s0, loss_s1, loss_all):
        lam_temp = torch.zeros(2)
        lam_temp[0] = client.lam[0] + self.eta*torch.abs(loss_all.mean() - loss_s0)
        lam_temp[1] = client.lam[1] + self.eta*torch.abs(loss_all.mean() - loss_s1)
        client.lam = lam_temp


