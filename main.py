from env import FederatedLearningEnvironment
from models import resnet18
from fedProtocol import *
from Losses import *

from torchvision import transforms
import torch
import timm

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

from functools import partial
import os
import yaml

FED_PROTOCOL = {"FedAvg": FedAvg,
                "FairFedAvgALM": FairFedAvgALM,
                "FedAvgFairALM": FedAvgFairALM,
                "FairFed": FairFed,
                "FPFL": FPFL}

EVAL_METHOD = {"CelebA": "one_client",
               "imsitu": "all_client"}

def load_config():
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
    return config

def train_instance(config, train_transforms, test_transforms, model_fn_fn, loss_fn, optimizer_fn_fn, federated_protocol_fn, lr_scheduler_fn_fn, 
    verbose=False, dataset_dir='dataset', checkpoint_dir=None, enforce_same_model_config=False, eval_metrics="normal", frac_each_client=1.0, **client_kwargs):
    def replace_default_with_sample(particular_config, all_config):
        for key in particular_config:
            if key in all_config:
                particular_config[key] = all_config[key]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    environmentFL = FederatedLearningEnvironment(n_clients = config["number_of_clients"], communication_round = config["maximum_communication_round"], seed = config["seed"],
    train_ratio = config["train_ratio"], dataset_dir = dataset_dir, device=device)

    #Use sampled value instead of the default
    replace_default_with_sample(config["optimizer_config"], config)
    replace_default_with_sample(config["model_config_client"], config)
    replace_default_with_sample(config["model_config_server"], config)
    replace_default_with_sample(config["fed_config"], config)
    replace_default_with_sample(config["lr_scheduler_config"], config)
    
    optimizer_fn = optimizer_fn_fn(config["optimizer_config"])
    model_fn_client = model_fn_fn(config['model_client'], config["model_config_client"])
    if enforce_same_model_config:
        model_fn_server = model_fn_fn(config['model_client'], config['model_config_client'])
    else:
        model_fn_server = model_fn_fn(config['model_server'], config['model_config_server'])
    federated_protocol = federated_protocol_fn(config['fed_config'])
    lr_scheduler_fn = lr_scheduler_fn_fn(config['lr_scheduler_config'])

    environmentFL.start_training(dataset_names = config["dataset_names"], train_transforms=train_transforms, test_transforms=test_transforms,
    dataset_options=config["dataset_option"], verbose=verbose, batch_size=config["batch_size"], dataset_config=config["dataset_config"],
    batch_config=config["loader_config"], model_fn_client=model_fn_client, model_fn_server=model_fn_server, loss_fn=loss_fn, optimizer_fn=optimizer_fn, fedProtocol=federated_protocol, local_epoch=config["local_epoch"], 
    checkpoint_dir=checkpoint_dir, lr_scheduler_fn=lr_scheduler_fn, use_ray=True, participating_ratio=config["participating_ratio"],
    eval_metrics=eval_metrics, frac_each_client=frac_each_client, eval_method=config["eval_method"], **client_kwargs)

def concat_dict(a, b):
    return {**a, **b}

if __name__  == '__main__':
    ray.init(include_dashboard=True)
    yaml_config = load_config()
    config = {}
    #Experiment configs
    EXPERIMENT_NAME = yaml_config["experiment_name"] #tensorboard ray_results/EXPERIMENT_NAME to visualize the result
    CHECKPOINT_DIR = yaml_config["checkpoint_dir"]
    GRACE_PERIOD = yaml_config["fed_config"]["communication_rounds"] #Period of halving
    NUM_OF_SAMPLES = 1
    CPU = yaml_config["cpu"]
    GPU = yaml_config["gpu"]
    STOPPER = TrialPlateauStopper(metric="eval_loss", std=yaml_config["stopper"]["std"], num_results=yaml_config["stopper"]["num_results"], grace_period=yaml_config["stopper"]["grace_period"])

    #FL environment-related configs
    N_CLIENTS = yaml_config["fed_config"]["n_clients"]
    COMMUNICATION_ROUND = GRACE_PERIOD
    SEED = yaml_config["seed"]
    evalMethod = EVAL_METHOD[yaml_config["dataset_config"]["dataset_name"]]
    if yaml_config["iterables"]["seed"]:
        config = concat_dict(config, {'seed': tune.grid_search(yaml_config["iterables"]["seed"])})
    if evalMethod == 'one_client':
        PARTICIPATING_RATIO = (N_CLIENTS - 1.0)/N_CLIENTS
        TRAIN_RATIO = 0.8
    else:
        PARTICIPATING_RATIO = 1.0
        TRAIN_RATIO = 0.8
    FRAC_EACH_CLIENT = yaml_config["fed_config"]["frac_num_local_samples_from_central"]

    #Fed Protocol configs
    if yaml_config["fed_protocol_instance"] not in FED_PROTOCOL:
        raise ValueError("Invalid federated algorithm!")
    FED_CONFIG = yaml_config[yaml_config["fed_protocol_instance"]]
    if yaml_config["iterables"]["beta"]:
        config = concat_dict(config, {'beta': tune.grid_search(yaml_config["iterables"]["beta"])})
    
    if yaml_config["iterables"]["sigma_dp"]:
        config = concat_dict(config, {'sigma_dp': tune.grid_search(yaml_config["iterables"]["sigma_dp"])})
    federated_protocol = FED_PROTOCOL[yaml_config["fed_protocol_instance"]]
    federated_protocol_fn = lambda c: federated_protocol(**c)
    EVAL_METRICS = "legal"

    DATASET_DIR = yaml_config["dataset_config"]["dataset_dir"]
    DATASET_NAMES = [yaml_config["dataset_config"]["dataset_name"]]
    if DATASET_NAMES[0] not in ["CelebA", "imsitu"]:
        raise ValueError("Only <CelebA> or <imsitu> argument is valid in dataset_name!")
    DATASET_OPTIONS = {'method': 'distribution', 'niid': 'dirichlet balanced'}
    if DATASET_NAMES[0] == "CelebA":
        TRAIN_TRANSFORM_DICT = {'CelebA': transforms.Compose([transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128, 128)),
                                            transforms.ToTensor()])}
        model_client = resnet18
        model_server = resnet18
        model_config_client = {'num_classes': 2, 'grayscale':False}
        model_config_server = {'num_classes': 2, 'grayscale':False}
    elif DATASET_NAMES[0] == "imsitu":
        TRAIN_TRANSFORM_DICT = {'imsitu': transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])}
        model_client = partial(timm.create_model, model_name="resnet18")
        model_server = partial(timm.create_model, model_name="resnet18")
        model_config_client = {'num_classes': 211, 'pretrained':True}
        model_config_server = {'num_classes': 211, 'pretrained':True}
    TEST_TRANSFORM_DICT = TRAIN_TRANSFORM_DICT
    VERBOSE = False
    DATASET_CONFIG = {'alpha': yaml_config["dataset_config"]["alpha"], 'label_attr': yaml_config["dataset_config"]["label_attr"], 'protected_attr': yaml_config["dataset_config"]["protected_attr"]}
    if yaml_config["iterables"]["alpha"]:
        config = concat_dict(config, {'alpha': tune.grid_search(yaml_config["iterables"]["alpha"])})
    BATCH_SIZE = yaml_config["dataset_config"]["batch_size"]
    LOADER_CONFIG = None

    #Basic model-related configs
    model_fn_fn = lambda m, l: lambda: m(**l)
    ENFORCE_SAME_MODEL_CONFIG = True

    loss_fn = torch.nn.CrossEntropyLoss()

    LOCAL_EPOCH = 1
    optimizer = torch.optim.SGD
    optimizer_fn_fn = lambda c: lambda x: optimizer(x, **c)
    optimizer_config = {'lr' : yaml_config["optimizer_config"]["learning_rate"]}
    if yaml_config["iterables"]["lr"]:
        config = concat_dict(config, {'lr': tune.grid_search(yaml_config["iterables"]["lr"])})
    lr_scheduler_config = {'step_size': yaml_config["optimizer_config"]["step_size"], 'gamma':yaml_config["optimizer_config"]["gamma"]}
    lr_scheduler_fn_fn = lambda c: lambda o: torch.optim.lr_scheduler.StepLR(o, **c)
    

    config_basic = {'number_of_clients': N_CLIENTS,
             'maximum_communication_round': COMMUNICATION_ROUND,
             'seed': SEED,
             'train_ratio': TRAIN_RATIO,
             'participating_ratio': PARTICIPATING_RATIO,
             'fed_config': FED_CONFIG, 
             'fed_protocol': federated_protocol,
             'dataset_names': DATASET_NAMES, 
             'dataset_option': DATASET_OPTIONS,
             'dataset_config': DATASET_CONFIG,
             'loader_config': LOADER_CONFIG,
             'batch_size': BATCH_SIZE,
             'model_config_client': model_config_client,
             'model_config_server': model_config_server,
             'optimizer_config': optimizer_config,
             'lr_scheduler_config': lr_scheduler_config,
             'local_epoch': LOCAL_EPOCH,
             'model_client': model_client,
             'model_server': model_server,
             'loss_fn': type(loss_fn).__name__,
             'optimizer_name': optimizer,
             'eval_method': evalMethod,
             }

    config = concat_dict(config_basic, config)

    scheduler = ASHAScheduler(
        max_t=COMMUNICATION_ROUND,
        grace_period=GRACE_PERIOD,
        reduction_factor=2)

    analysis = tune.run(
        partial(train_instance, train_transforms=TRAIN_TRANSFORM_DICT, test_transforms=TEST_TRANSFORM_DICT, model_fn_fn=model_fn_fn, 
        loss_fn=loss_fn, optimizer_fn_fn=optimizer_fn_fn, federated_protocol_fn=federated_protocol_fn, verbose=VERBOSE, dataset_dir=DATASET_DIR, 
        checkpoint_dir=CHECKPOINT_DIR, lr_scheduler_fn_fn=lr_scheduler_fn_fn,
        enforce_same_model_config=ENFORCE_SAME_MODEL_CONFIG, eval_metrics=EVAL_METRICS, frac_each_client=FRAC_EACH_CLIENT,
        **{'pos_label':yaml_config['dataset_config']['pos_label'], 'neg_label': yaml_config['dataset_config']['neg_label']}),
        resources_per_trial = {"cpu": CPU, "gpu": GPU},
        config=config,
        name = EXPERIMENT_NAME,
        num_samples=NUM_OF_SAMPLES,
        scheduler=scheduler,
        metric="eval_loss",
        mode="min",
        local_dir="./ray_results",
        resume=False,
        stop=STOPPER
    )
    
    