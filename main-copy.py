from env import FederatedLearningEnvironment
from models import LeNet, resnet18
from fedProtocol import *
from optimizer import *
from communication_channel import *
from Losses import *
from lr_schedule import *
from postprocess import *

from torchvision import transforms
import torch
import timm
from timm.data.transforms_factory import create_transform

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune import CLIReporter

from functools import partial
import sys


def train_instance(config, train_transforms, test_transforms, model_fn_fn, loss_fn, optimizer_fn_fn, federated_protocol_fn, lr_scheduler_fn_fn, 
    communication_channel_fn, postprocess_list = [], verbose=False, dataset_dir='dataset', checkpoint_dir=None, enforce_same_model_config=False, fine_tune=False,
    eval_metrics="normal", frac_each_client=1.0):
    def replace_default_with_sample(particular_config, all_config):
        for key in particular_config:
            if key in all_config:
                particular_config[key] = all_config[key]

    from models import vpt_tiny_patch16_384, vpt_small_patch16_224
    device = "cuda" if torch.cuda.is_available() else "cpu"

    environmentFL = FederatedLearningEnvironment(n_clients = config["number_of_clients"], communication_round = config["maximum_communication_round"], seed = config["seed"],
    train_ratio = config["train_ratio"], dataset_dir = dataset_dir, device=device)

    #Use sampled value instead of the default
    replace_default_with_sample(config["optimizer_config"], config)
    replace_default_with_sample(config["model_config_client"], config)
    replace_default_with_sample(config["model_config_server"], config)
    replace_default_with_sample(config["fed_config"], config)
    replace_default_with_sample(config["lr_scheduler_config"], config)
    replace_default_with_sample(config["communication_channel_config"], config)
    
    optimizer_fn = optimizer_fn_fn(config["optimizer_config"])
    model_fn_client = model_fn_fn(config['model_client'], config["model_config_client"])
    if enforce_same_model_config:
        model_fn_server = model_fn_fn(config['model_client'], config['model_config_client'])
    else:
        model_fn_server = model_fn_fn(config['model_server'], config['model_config_server'])
    federated_protocol = federated_protocol_fn(config['fed_config'])
    lr_scheduler_fn = lr_scheduler_fn_fn(config['lr_scheduler_config'])
    communication_channel = communication_channel_fn(config['communication_channel_config'])

    environmentFL.start_training(dataset_names = config["dataset_names"], train_transforms=train_transforms, test_transforms=test_transforms,
    dataset_options=config["dataset_option"], verbose=verbose, batch_size=config["batch_size"], dataset_config=config["dataset_config"],
    batch_config=config["loader_config"], model_fn_client=model_fn_client, model_fn_server=model_fn_server, loss_fn=loss_fn, optimizer_fn=optimizer_fn, fedProtocol=federated_protocol, local_epoch=config["local_epoch"], 
    checkpoint_dir=checkpoint_dir, lr_scheduler_fn=lr_scheduler_fn, use_ray=True, communication_channel=communication_channel, participating_ratio=config["participating_ratio"], green_mode=config["green_mode"], postProcessList=postprocess_list,
    fine_tune=fine_tune, eval_metrics=eval_metrics, frac_each_client=frac_each_client)

def concat_dict(a, b):
    return {**a, **b}

if __name__  == '__main__':
    '''
    If the search space is contained in a dictionary you need to 
    write the variable again in dictionary and concat with the config.
    Treat the existing as the default value
    Example:
    optimizer_config = {'lr': 1e-2}
    LRs = {'lr': tune.uniform(1e-2, 1e-1)}
    config = concat_dict(config, LRs)

    PLEASE DONT FORGET TO PUT THE KEY PARAMETER ON THE DEFAULT CONFIG OR OTHERWISE THE SCAN WILL NOT WORK

    Current support are:
    - optimizer_config
    - model_config for client and server
    - fed_config
    - model config on both sides
    - global state config
    - learning rate decay config
    '''
    ray.init(include_dashboard=True)
    config = {}
    #Experiment configs
    EXPERIMENT_NAME = "imsitu_FPFL_4clients_alpha2_lessless" #tensorboard ray_results/EXPERIMENT_NAME to visualize the result
    CHECKPOINT_DIR = None#"../train_instance_5e95c_00000_0_2023-03-02_12-26-15/model_checkpoint/global_model round = 40.pt"
    GRACE_PERIOD = 200 #Period of halving
    NUM_OF_SAMPLES = 1 #If grid_search, it means number of realization. Otherwise, it is literally number of sample
    CPU = 7 #1-16
    GPU = 0.5 #(0.0,1.0]
    RESUME = False
    STOPPER = TrialPlateauStopper(metric="eval_loss", std=1e-6, num_results=8, grace_period=20)

    #FL environment-related configs
    N_CLIENTS = 4
    COMMUNICATION_ROUND = GRACE_PERIOD
    SEED = 3
    PARTICIPATING_RATIO = 1. #For the whole training
    GREEN_MODE = False
    TRAIN_RATIO = 0.8 #train-val proportion on client side
    FRAC_EACH_CLIENT = 1.0 #Control the number of data per client in case too excessive

    COMMUNICATION_CHANNEL = None#BinaryCommunicationChannel#QSGDCommunicationChannel
    COMMUNICATION_CHANNEL_FN = lambda c: None#COMMUNICATION_CHANNEL(**c)
    COMMUNICATION_CHANNEL_CONFIG = {}
    #COMMUNICATION_CHANNEL_CONFIG = {'quantization_level': 8, 'BER': 0, 'clip_val': 5}
    #config = concat_dict(config, {'quantization_level': tune.choice([2**2, 2**4, 2**8, 2**16]), 'BER': tune.choice([1e-2, 1e-3, 1e-4])})

    #Fed Protocol configs
    #FED_CONFIG = {"protocol": FedAvg(C=1.0), "num_generated_data": 200, "batch_size": 32, "server_epoch": 10}
    FED_CONFIG = {
    'C': 1.0, 
    'grad_clip': 1.0,  
    'beta': 0.1, 
    'eta': 0.2
  }
    config = concat_dict(config, {'seed': tune.grid_search([1,2,3])})
    # config = concat_dict(config, {'beta': tune.grid_search([0.1,0.3,0.5,0.75])})
    federated_protocol = FPFL
    federated_protocol_fn = lambda c: federated_protocol(**c)
    FINE_TUNE = False
    EVAL_METRICS = "legal"
    #config = concat_dict(config, {'lr_g': tune.uniform(0.1, 1.0)})

    #dataset configs
    DATASET_DIR = "../../../../dataset/"
    DATASET_NAMES = ["imsitu"]
    DATASET_OPTIONS = {'method': 'distribution', 'niid': 'dirichlet balanced'}
    # DATASET_OPTIONS = {'method': 'semantic', 'niid': 'gmm-cv-classification'}
    # DATASET_OPTIONS = {'method': 'ViT-FL_paper_cifar10', 'niid': 'split_3'}
    #TRAIN_TRANSFORM_DICT = {'CIFAR10': create_transform(224,)}
    # TRAIN_TRANSFORM_DICT =  {'CIFAR10': transforms.Compose([
    #                             transforms.RandomCrop(32, padding=4),
    #                             #transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                         ])}
    TRAIN_TRANSFORM_DICT = {'imsitu': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])}
    # TRAIN_TRANSFORM_DICT = {'CIFAR100': transforms.Compose([
    #             transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])}
    # TRAIN_TRANSFORM_DICT = {'OCT2017':transforms.Compose([
    #             transforms.RandomResizedCrop((224,224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
    # TRAIN_TRANSFORM_DICT = train_transform = {'MNIST': transforms.Compose([
    #                                                     transforms.ToTensor(),
    #                                                     transforms.Normalize((0.5), (0.5))
    #                                                 ])}
    # TEST_TRANSFORM_DICT = TRAIN_TRANSFORM_DICT
    # TEST_TRANSFORM_DICT = {'CIFAR100': transforms.Compose([
    #             transforms.Resize((224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])}
    TEST_TRANSFORM_DICT = TRAIN_TRANSFORM_DICT
    VERBOSE = False
    DATASET_CONFIG = {'alpha': 2.0}
    BATCH_SIZE = 128
    LOADER_CONFIG = None

    #Basic DL-related configs
    # model_client = lambda: LeNet()
    model_client = partial(timm.create_model, model_name="resnet18")
    # model_client = ResNet14
    model_fn_fn = lambda m, l: lambda: m(**l)
    # model_config_client = {'num_classes': 10, 'in_channel': 3}
    model_config_client = {'num_classes': 211, 'pretrained':True}#, 'num_prompt_tokens_per_layer':1, 'mode':'deep', 'prompt_depth':4}

    model_server = partial(timm.create_model, model_name="resnet18")#lambda: model_client = partial(timm.create_model, model_name="resnet18")
    # model_server = partial(timm.create_model, model_name="vit_lora_small_patch16_224")
    # model_server = ResNet14
    # model_config_server = {'num_classes': 10, 'in_channel': 3}
    model_config_server = {'num_classes': 211, 'pretrained':True}
    # config = concat_dict(config, {'down_sample': tune.grid_search([4,16,64])})
    ENFORCE_SAME_MODEL_CONFIG = True

    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = [torch.nn.CrossEntropyLoss, FocalLoss]
    # loss_config = [{}, {'gamma': 0.5}]
    # loss_lambda = lambda l_fn, c : l_fn(**c) 

    LOCAL_EPOCH = 1
    optimizer = torch.optim.SGD
    optimizer_fn_fn = lambda c: lambda x: optimizer(x, **c)
    optimizer_config = {'lr' : 0.02}#, 'weight_decay':1e-5}
    # config = concat_dict(config, {'lr': tune.grid_search([5e-3, 1e-3, 5e-4, 1e-4, 5e-5])})
    #lr_scheduler_config = {'warmup_steps':500, 't_total':50*COMMUNICATION_ROUND*LOCAL_EPOCH} #Hard-coded number is estimated number of iterations per round per client oct:6000/32 cifar100:93 cifar10:63
    lr_scheduler_config = {'step_size': 40, 'gamma':0.5}
    #config = concat_dict(config, {'step_size': tune.choice([200, 400, 1000]), 'gamma':tune.choice([0.95, 0.9, 0.1])})
    lr_scheduler_fn_fn = lambda c: lambda o: torch.optim.lr_scheduler.StepLR(o, **c)
    #lr_scheduler_fn_fn = lambda c: lambda o: None
    

    # POSTPROCESS_LIST = [WeightDivergence(model_class=partial(timm.create_model, model_name="vit_small_patch16_224"), model_args={'num_classes': 10, 'pretrained': False}, centralized_dir="centralized_models/vit-small-cifar10-scratch.pt", filename_base="testing"),
    #                 CosineSimilarityOfPosEmbeddings(filename_base="testing")]
    POSTPROCESS_LIST = []#[CosineSimilarityOfPromptEmbeddings(filename_base="one")]

    config_basic = {'number_of_clients': N_CLIENTS,
             'maximum_communication_round': COMMUNICATION_ROUND,
             'seed': SEED,
             'train_ratio': TRAIN_RATIO,
             'participating_ratio': PARTICIPATING_RATIO,
             'green_mode': GREEN_MODE,
             'fed_config': FED_CONFIG, 
             'fed_protocol': federated_protocol,
             'dataset_names': DATASET_NAMES, 
             'dataset_option': DATASET_OPTIONS,
             'dataset_config': DATASET_CONFIG,
             'loader_config': LOADER_CONFIG,
             'communication_channel_config': COMMUNICATION_CHANNEL_CONFIG,
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
             }

    config = concat_dict(config_basic, config) #Existing config override the base config

    scheduler = ASHAScheduler(
        max_t=COMMUNICATION_ROUND,
        grace_period=GRACE_PERIOD,
        reduction_factor=2)

    analysis = tune.run(
        partial(train_instance, train_transforms=TRAIN_TRANSFORM_DICT, test_transforms=TEST_TRANSFORM_DICT, model_fn_fn=model_fn_fn, 
        loss_fn=loss_fn, optimizer_fn_fn=optimizer_fn_fn, federated_protocol_fn=federated_protocol_fn, verbose=VERBOSE, dataset_dir=DATASET_DIR, 
        checkpoint_dir=CHECKPOINT_DIR, lr_scheduler_fn_fn=lr_scheduler_fn_fn, communication_channel_fn=COMMUNICATION_CHANNEL_FN, postprocess_list=POSTPROCESS_LIST,
        enforce_same_model_config=ENFORCE_SAME_MODEL_CONFIG, fine_tune=FINE_TUNE, eval_metrics=EVAL_METRICS),
        resources_per_trial = {"cpu": CPU, "gpu": GPU},
        config=config,
        name = EXPERIMENT_NAME,
        num_samples=NUM_OF_SAMPLES,
        scheduler=scheduler,
        metric="eval_loss",
        mode="min",
        local_dir="./ray_results",
        resume=RESUME,
        stop=STOPPER
    )
    
    