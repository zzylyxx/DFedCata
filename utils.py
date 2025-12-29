import torch
import numpy as np


def get_mdl_params(model):
    '''
    Flatten all learnable parameters in the model into a one-dimensional vector. This vector is stored separately and is independent of parameter updates in the model.
    .clone() indicates deep copy
    '''
    # model parameters ---> vector (different storage)
    vec = []
    for param in model.parameters():
        vec.append(param.clone().detach().cpu().reshape(-1))
    return torch.cat(vec)

def get_mdl_params_2(model):
    '''
    Flatten all learnable parameters in the model into a one-dimensional vector. This vector is stored separately and is independent of parameter updates in the model.
    .clone() indicates deep copy
    '''
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)



def param_to_vector(model):
    '''
    Flatten all learnable parameters in the model into a one-dimensional vector, but this vector still shares storage space with the model's parameters.
    Advantage: model parameters can be updated by updating values in the vector
    '''
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def grad_param_to_vector(model):
    '''
    Flatten all learnable parameters in the model into a one-dimensional vector, but this vector still shares storage space with the model's parameters.
    Advantage: model parameters can be updated by updating values in the vector
    '''
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.grad.reshape(-1))
    return torch.cat(vec)


def set_client_from_params(device, model, params):
    '''
    Based on values in params (which is one-dimensional), this function reshapes them to the required size in the model, then shallow copies and replaces the original parameters in the model,
    finally returns the model with new parameters from params
    '''
    idx = 0
    for param in model.parameters():
        length = param.numel()
        param.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)

def grad_set_client_from_params(device, model, params):
    '''
    Based on values in params (which is one-dimensional), this function reshapes them to the required size in the model, then shallow copies and replaces the original parameters in the model,
    finally returns the model with new parameters from params
    '''
    idx = 0
    for param in model.parameters():
        length = param.grad.numel()
        param.grad.data.copy_(params[idx:idx + length].reshape(param.shape))
        idx += length
    return model.to(device)


def get_params_list_with_shape(model, param_list):
    '''
    Reorganize data from one-dimensional list `param_list` into a list of tensors that match the model parameter shapes, and return it
    '''
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape

def generate_P(mode, size):
    '''
    Generate communication topology based on name
    mode:str "all" "single" "star" "meshgrid" "exponential"
    size:int
    '''
    result = torch.zeros((size, size))
    if mode == "all":
        result = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            result[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            result[i][i] = 1 / 3
            result[i][(i - 1 + size) % size] = 1 / 3
            result[i][(i + 1) % size] = 1 / 3
    elif mode == "star":
        for i in range(size):
            result[i][i] = 1 - 1 / size
            result[0][i] = 1 / size
            result[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(len(topo_neighbor_with_self[i]),
                                           len(topo_neighbor_with_self[j]))
            topo[i][i] = 2.0 - topo[i].sum()
        result = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        result = torch.tensor(topo, dtype=torch.float)
    return result