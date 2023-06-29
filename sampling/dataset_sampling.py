import numpy as np
import collections

def get_iid_data(train_idcs, n_clients):
    return np.array_split(train_idcs, n_clients)

def get_niid_dirichlet_unbalanced(train_idcs, train_labels, alpha, n_clients):
    '''
    This function breaks if train_labels is not in the form of [N-1]
    TO DO: support any form of labels
    '''
    n_classes = int(train_labels.max())+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes) #Distribution related to number of dataset hold by each client

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))): #Given a class label extract the index class with corresponding probability of each device
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs] #After manipulating indices  load the indices mapping to the data
    
    return client_idcs

def sinkhorn(A,  L):
    N = A.shape[0]
    M = A.shape[1]
    for i in range(L):
        A = A / np.matmul(np.ones((1, N)), A)
        A = A / np.matmul(A, np.ones((M, 1)))
    return A

def get_niid_dirichlet_balanced(train_labels, alpha, n_clients, frac_from_full):
    '''
    This function breaks if train_labels is not [N-1]
    TO DO: support any form of labels
    '''
    n_classes = int(train_labels.max())+1

    #Find label with less samples
    freq_dict = collections.Counter(train_labels)
    m = min(freq_dict, key=freq_dict.get)
    _lst_sample = freq_dict[m] 

    #Trim according to the number above for each label
    idcs_per_class = np.zeros((n_classes, _lst_sample), dtype=int)
    for i in range(n_classes):
        idcs_per_class[i, :] = np.random.choice(np.where(train_labels==i)[0], _lst_sample, replace=False)

    #Get indices for each label and assign proportion of samples according to Dirichlet distribution
    idcs_per_class = [list(ipc) for ipc in idcs_per_class]
    distribution = np.random.dirichlet(np.full(shape=n_clients, fill_value=alpha),n_classes)
    distribution = sinkhorn(distribution, 100)

    idcs_per_client = [[] for _ in range(n_clients)]

    n_data = n_classes*len(idcs_per_class[0])
    budget_per_client = n_data/n_clients
    for i in range(n_classes):
        proportion = np.cumsum(distribution[i]*len(idcs_per_class[i])).astype(int)
        proportion = np.insert(proportion, 0, 0)
        for j in range(n_clients):
            idcs_per_client[j].extend(idcs_per_class[i][proportion[j]:proportion[j+1]])

    first_idcs = [np.array(data).astype(int) for data in idcs_per_client]
    num_per_client = int(min([len(idcs) for idcs in first_idcs])*frac_from_full)
    for i in range(n_clients):
        first_idcs[i] = np.random.choice(first_idcs[i], num_per_client, replace=False)
    return first_idcs


