from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from torchvision import transforms

import numpy as np
from typing import Any, List, Dict, Union
from filelock import FileLock
import os

from sampling.dataset_sampling import *
from dataset import *

class DatasetManager():
    """
    This class will produce datasets used for the server and the clients. All dataset preprocessing are handled here
    How to use:
    1. Instatiate this class
    2. Call load_dataset()
    3. Before distribute the data, dont forget to call set_transform(). Call augment_data() if necessary
    4. Call distribute_data to get the dataset objects for the clients and the server
    5. Call get_dataloader()
    """
    def __init__(self, n_clients: int, train_transform_dict: Dict[str, transforms.Compose], test_transform_dict: Dict[str, transforms.Compose], 
                options: Dict[str, Any], verbose:bool = False, params: Dict[str, Any] = {}, train_ratio = 0.8, use_ray = False, frac_each_client = 1.0) -> None:
        """
        If the transform is not specified, the dataset will only be converted to Tensor
        """
        self.n_clients = n_clients
        self.train_transform_dict = train_transform_dict
        self.test_transform_dict = test_transform_dict
        self.options = options
        self.verbose = verbose
        self.params = params
        self.split_before_distribute = True

        if options["method"] == "semantic":
            self.split_before_distribute = False
        
        self.train_ratio = train_ratio
        self.use_ray = use_ray
        self.frac_each_client = frac_each_client
    
    def _split_dataset(self, data: Union[Dataset, List[Dataset]]) -> None: #Split according to the specified ratio
        if isinstance(data, list):
            data = ConcatDataset(data)
        train_size = int(self.train_ratio*len(data))
        val_size = len(data) - train_size
        return random_split(data, [train_size, val_size])

    def load_and_distribute_dataset(self, dataset_names: List[str], dataset_dir: str="dataset")-> Union[Any, Any]:
        """
        params:
        - train_size: size of training dataset
        - val_size: size of validation dataset

        The size of test dataset is inferred. The user should know the dataset size
        """
        self.dataset_names = dataset_names
        self.train_data = {}
        self.val_data = {}
        self.data = {}

        with FileLock(os.path.expanduser("~/data.lock")): #Data write or download is not thread-safe
            if "CelebA" in dataset_names:
                label_lower = self.params["label_attr"].lower()
                train_data = CelebaDataset(csv_path = dataset_dir + f"CelebA/celeba-{label_lower}-train.csv",
                                           img_dir = dataset_dir + 'CelebA/img_align_celeba/',
                                           label_attr = self.params["label_attr"],
                                           protected_attr = self.params["protected_attr"])
                val_data = CelebaDataset(csv_path = dataset_dir + f'CelebA/celeba-{label_lower}-valid.csv',
                                  img_dir = dataset_dir + 'CelebA/img_align_celeba/',
                                  label_attr = self.params["label_attr"],
                                  protected_attr = self.params["protected_attr"])
                test_data = CelebaDataset(csv_path=dataset_dir + f'CelebA/celeba-{label_lower}-test.csv',
                                 img_dir = dataset_dir + 'CelebA/img_align_celeba/',
                                 label_attr = self.params["label_attr"],
                                  protected_attr = self.params["protected_attr"])
                if self.split_before_distribute:
                    self.train_data["CelebA"], self.val_data["CelebA"] = train_data, val_data
                else:
                    self.data["CelebA"] = ConcatDataset([train_data, val_data, test_data])
                
            if "imsitu" in dataset_names:
                train_data = ImSituVerbGender(dataset_dir=dataset_dir, annotation_dir="metadata", image_dir = "of500_images_resized", split="train")
                val_data = ImSituVerbGender(dataset_dir=dataset_dir, annotation_dir="metadata", image_dir = "of500_images_resized", split="val")
                test_data = ImSituVerbGender(dataset_dir=dataset_dir, annotation_dir="metadata", image_dir = "of500_images_resized", split="test")
                if self.split_before_distribute:
                    self.train_data["imsitu"], self.val_data["imsitu"] = train_data, val_data
                else:
                    self.data["imsitu"] = ConcatDataset([train_data, val_data, test_data])
                
        return self.distribute_data()
    
    def distribute_data(self) -> Union[Any, Any]:
        """
        params:
        options consists of
            - method : either 'chunk' or 'distribution'.
            - niid : specify how to simulate non-iid. For distribution, 'dirichlet unbalanced' or 'dirichlet balanced' can be used
                     , while for chunks, it can be 'domain' which simulates heterogeneity (feature shift) or 'label' which creates shards
            - client_eval: specify whether the model testing is carried out by clients or server

        verbose return a handful statistics of clients data

        return:
            Subsets for clients and server
        """
        assert 'method' in self.options
        assert 'niid' in self.options
        client_data_list = []
        val_data_list = []
        if self.options['method'] == 'distribution':
            self._set_transform()
            if self.options['niid'] == 'dirichlet unbalanced':
                for dataset_name in self.dataset_names:
                    train_labels = np.array([self.train_data[dataset_name][idx][1] for idx in range(len(self.train_data[dataset_name]))])
                    train_idcs = np.random.permutation(len(self.train_data[dataset_name]))
                    client_idcs = get_niid_dirichlet_unbalanced(train_idcs, train_labels, self.params['alpha'], self.n_clients)
                    if self.verbose:
                        print("Minimum number of data possessed by a client is " + str(min([len(x) for x in client_idcs])))
                        print("Maximum number of data possessed by a client is " + str(max([len(x) for x in client_idcs])))
                    client_data_list.append([Subset(self.train_data[dataset_name], client_idx) for client_idx in client_idcs])

            elif self.options['niid'] == 'dirichlet balanced':
                for dataset_name in self.dataset_names:
                    train_labels = np.array([self.train_data[dataset_name][idx][1] for idx in range(len(self.train_data[dataset_name]))])
                    client_idcs = get_niid_dirichlet_balanced(train_labels, self.params['alpha'], self.n_clients, self.frac_each_client)
                    if self.verbose:
                        print("Minimum number of data possessed by a client is " + str(min([len(x) for x in client_idcs])))
                        print("Maximum number of data possessed by a client is " + str(max([len(x) for x in client_idcs])))
                    client_data_list.append([Subset(self.train_data[dataset_name], client_idx) for client_idx in client_idcs])     
            else:
                raise ValueError("The distribution does not support the current niid method!")
        else:
            raise ValueError(f"The {self.options['method']} method is not supported!")
        for dataset_name in self.dataset_names:
            val_idcs = np.arange(len(self.val_data[dataset_name]))
            client_val_idcs = get_iid_data(val_idcs, n_clients=self.n_clients)
            val_data_list.append([Subset(self.val_data[dataset_name], idx) for idx in client_val_idcs])
        val_data_temp = list(map(list, zip(*val_data_list)))
        val_data = [ConcatDataset(c) for c in val_data_temp]
        client_data_temp = list(map(list, zip(*client_data_list))) #Transpose operation
        client_data = [ConcatDataset(c) for c in client_data_temp]
        
        return client_data, val_data
    
    def _set_transform_train(self):
        default_transform = None#transforms.Compose([transforms.ToTensor()])
        for dataset_name in self.dataset_names:
            tr_train = self.train_transform_dict[dataset_name] if dataset_name in self.train_transform_dict else default_transform
            self.train_data[dataset_name] = DatasetWrapper(self.train_data[dataset_name], tr_train)

    def _set_transform_val(self):
        default_transform = None#transforms.Compose([transforms.ToTensor()])
        for dataset_name in self.dataset_names:
            tr_test = self.test_transform_dict[dataset_name] if dataset_name in self.test_transform_dict else default_transform
            self.val_data[dataset_name] = DatasetWrapper(self.val_data[dataset_name], tr_test)

    def _set_transform(self):
        self._set_transform_train()
        self._set_transform_val()

    def get_dataloader(self, data: Any, batch_size: int, options: Dict[str, Any] = {}):
        '''
        Configure dataloader here for modularity
        data can be single DataLoader or List of DataLoaders
        '''
        if not options:
            if isinstance(data, list):
                dataloader_list = []
                for d in data:
                    dataloader_list.append(DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=False))
                return dataloader_list
            else:
                return DataLoader(data, batch_size=batch_size, shuffle=True)
        else:
            raise NotImplementedError()
