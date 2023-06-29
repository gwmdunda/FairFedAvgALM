import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from itertools import chain
from typing import Any, List, Callable, Dict

import fedProtocol 

class FederatedNode(object):
    def __init__(self, model_fn: Callable[[], torch.nn.Module], fedProtocol: fedProtocol.FedProtocol,
    train_loader: DataLoader = None, eval_loader: DataLoader = None, loss_fn: Any = None, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer] = None, device = 'cpu') -> None:
        self.model = model_fn().to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer_fn(self.model.parameters()) if optimizer_fn is not None else None
        self.fedProtocol = fedProtocol
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.ensemble_model = None
    
    def eval(self, loader=None, **kwargs):
        if self.eval_loader is None and loader is None:
            return
        if loader is None:
            loader = self.eval_loader
        elif loader == "all":
            loader = chain(self.train_loader, self.eval_loader)
            l = len(self.train_loader.dataset) + len(self.eval_loader.dataset)
        elif loader == "train":
            loader = self.train_loader
            l = len(loader.dataset)
        else:
            l = len(loader.dataset)
        self.model.eval()

        eval_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                eval_loss += self.loss_fn(out, y).item()
                pred = out.data.max(1)[1]
                correct += pred.eq(y.view(-1)).sum().item()
        
        return eval_loss/l, correct/l


class Client(FederatedNode):
    def __init__(self, id: int, model_fn: Callable[[], torch.nn.Module], fedProtocol: fedProtocol.FedProtocol, train_loader: DataLoader = None,
    eval_loader: DataLoader = None, loss_fn: Any = None, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer] = None, 
    device='cpu', local_epoch: int = 1, lr_scheduler_fn: Callable[[torch.optim.Optimizer], _LRScheduler] = lambda _ : None) -> None:
        super().__init__(model_fn, fedProtocol, train_loader, eval_loader,  loss_fn, optimizer_fn, device)
        self.id = id
        self.local_epoch = local_epoch
        self.lr_schedule = lr_scheduler_fn(self.optimizer)
        self.lr_scheduler_fn = lr_scheduler_fn

    def local_compute(self, loader: DataLoader = None, lr_schedule: _LRScheduler = None):
        if loader is None:
            loader = self.train_loader
        if lr_schedule is None: #This is normal mode where client just use their learning rate schedule
            lr_schedule = self.lr_schedule
            has_lr_schedule = True
        elif isinstance(lr_schedule, str): #This is green mode, in which proxy clients holds the model and needs outside lr schedule
            lr_schedule = self.lr_schedule
            has_lr_schedule = False
        else: 
            lr_schedule = self.lr_schedule.load_state_dict(lr_schedule.state_dict())
            has_lr_schedule = False
        stats = self.fedProtocol.local_compute(self, loader, lr_schedule)
        if not has_lr_schedule: #Exclusive to green mode
            self.lr_schedule = self.lr_scheduler_fn(self.optimizer)
        return stats, lr_schedule

class Server(FederatedNode):
    def __init__(self, model_fn: Callable[[], torch.nn.Module], fedProtocol: fedProtocol.FedProtocol, train_loader: DataLoader = None, 
    eval_loader: DataLoader = None, loss_fn: Any = None, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer] = None, device='cpu') -> None:
        super().__init__(model_fn, fedProtocol, train_loader, eval_loader, loss_fn, optimizer_fn, device)

    def broadcast_weight(self, clients: List[Client], **kwargs):
        self.fedProtocol.broadcast_weight(clients, self.model, **kwargs)

    def aggregate_weights(self, clients: List[Client], global_state: Dict = {}):
        return self.fedProtocol.aggregation(self, clients, global_state)

    def select_clients(self, clients: List[Client]):
        return self.fedProtocol.select_client(clients)
    
    def post_processing(self, clients):
        return self.fedProtocol.post_processing(self, clients)

class LegalClient(Client):
    def __init__(self, id: int, model_fn: Callable[[], torch.nn.Module], fedProtocol: fedProtocol.FedProtocol, train_loader: DataLoader = None,
    eval_loader: DataLoader = None, loss_fn: Any = None, optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer] = None, 
    device='cpu', local_epoch: int = 1, lr_scheduler_fn: Callable[[torch.optim.Optimizer], _LRScheduler] = lambda _ : None, pos_label=1, neg_label=0) -> None:
        super().__init__(id, model_fn, fedProtocol, train_loader, eval_loader, loss_fn, optimizer_fn, device, local_epoch, lr_scheduler_fn)
        self.pos_label = pos_label
        self.neg_label = neg_label
    
    def eval(self, loader=None):
        if self.eval_loader is None and loader is None:
            return
        if loader is None:
            loader = self.eval_loader
        if loader == "all":
            loader = chain(self.train_loader, self.eval_loader)
            l = len(self.train_loader.dataset) + len(self.eval_loader.dataset)
        elif loader == "train":
            loader = self.train_loader
            l = len(loader.dataset)
        else:
            l = len(loader.dataset)
        self.model.eval()

        pos_label = self.pos_label
        neg_label = self.neg_label

        acc = 0.0
        num_pos_pred_protected0, num_pos_pred_protected1 = 0.0, 0.0
        num_pred1_targets1_protected0, num_pred1_targets1_protected1 = 0.0, 0.0
        count = 0.0
        count_pos = 0.0
        eval_loss = 0.0
        with torch.no_grad():
            for features, targets, protected in loader:
                features, targets, protected = features.to(self.device), targets.to(self.device), protected.to(self.device)
                out = self.model(features)
                eval_loss += self.loss_fn(out, targets).item()
                predicted_labels = out.data.max(1)[1]

                mask = ((targets == pos_label) | (targets == neg_label))
                count_pos += (targets == pos_label).float().sum().detach().cpu().item()
                count += mask.float().sum().detach().cpu().item()
                
                acc += (targets == predicted_labels).float().sum().detach().cpu().item()

                num_pos_pred_protected0 += (mask & (predicted_labels == pos_label) & (protected == 0)).float().sum().detach().cpu().item()
                num_pos_pred_protected1 += (mask & (predicted_labels == pos_label) & (protected == 1)).float().sum().detach().cpu().item()

                num_pred1_targets1_protected0 += (mask & (predicted_labels == pos_label) & (targets == pos_label) & (protected == 0)).float().sum().cpu().numpy()
                num_pred1_targets1_protected1 += (mask & (predicted_labels == pos_label) & (targets == pos_label) & (protected == 1)).float().sum().cpu().numpy()
        DP = abs(num_pos_pred_protected0 / (count + 1e-6) - num_pos_pred_protected1 / (count + 1e-6)) * 100
        EO = abs(num_pred1_targets1_protected0/(count_pos + 1e-6) - num_pred1_targets1_protected1/(count_pos + 1e-6))*100
        return eval_loss/l, acc*100./l, DP, EO

                
