"""
DFedCata Federated Learning Framework

This script implements the main training loop for federated learning experiments
using the DFedCata algorithm. It supports various datasets, models, and experimental
configurations for benchmarking federated learning performance.

Usage:
    python train_1.py --dataset CIFAR10 --model LeNet --total-client 100 --comm-rounds 500

Author: Your Name
"""

import torch
import argparse
import random
import os
import numpy as np

from utils import *
from models import *
from server import *
from dataset import *

#### ================= Open Float32 in A100 ================= ####
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#### ================= Open ignore warining ================= ####
import warnings
warnings.filterwarnings('ignore')
#### ======================================================== ####
print("##=============================================##")
print("##     Federated Learning Simulator Starts     ##")
print("##=============================================##")

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
# parser.add_argument('--mode', choices=['all', 'star','single','meshgrid','exponential'], type=str, default='all')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100','mnist','tinyimagenet'], type=str, default='CIFAR10')             # select dataset
parser.add_argument('--model', choices=['ResNet18', 'ResNet18_tinyimagenet','LeNet'], type=str, default='LeNet')                    # select model
parser.add_argument('--non-iid', action='store_true', default=True)                                       # activate if use heterogeneous dataset
parser.add_argument('--split-rule', choices=['Dirichlet', 'Path'], type=str, default='Dirichlet')          # select the dataset splitting rule ,Path is Pathological
parser.add_argument('--split-coef', default=0.3, type=float)                                                 # --> if Dirichlet: select the Dirichlet coefficient (i.e. 0.1, 0.3, 0.6, 1)

parser.add_argument("--cs", type=str, default='random')                                                                                                             # --> if Pathological: select the Dirichlet coefficient (i.e. 3, 5)
parser.add_argument('--active-ratio', default=0.1, type=float)                                              # select the partial participating ratio (i.e. 0.1, 0.05)
parser.add_argument('--total-client', default=100, type=int)                                               # select the total number of clients (i.e. 100, 500)
parser.add_argument('--comm-rounds', default=500, type=int)                                               # select the global communication rounds T
parser.add_argument('--local-epochs', default=5, type=int)                                                 # select the local interval K
parser.add_argument('--batchsize', default=128, type=int)                                                   # select the batchsize
parser.add_argument('--weight-decay', default=0.0005, type=float)                                           # select the weight-decay (i.e. 0.01, 0.001)
parser.add_argument('--local-learning-rate', default=0.1, type=float)                                      # select the local learning rate (generally 0.1 expect for local-adaptive-based)                                  
parser.add_argument('--lr-decay', default=0.998, type=float)                                               # select the learning rate decay (generally 0.998 expect for proxy-based)                                              
parser.add_argument('--seed', default=20, type=int)                                                        # select the random seed
parser.add_argument('--cuda', default=0, type=int)                                                         # select the cuda ID
parser.add_argument('--data-file', default='./', type=str)                                                 # select the path of the root of Dataset
parser.add_argument('--out-file', default='out/', type=str)                                                # select the path of the log files                                                

                                               
parser.add_argument('--mu', default=0.9, type=float)                                                    # select the coefficient for client-momentum
parser.add_argument('--beta', default=0.9, type=float)                                                     # select the coefficient for relaxed initialization                                                   
parser.add_argument('--rho', default=0, type=float)                                                      # select the SAM perturbation rate                                                
parser.add_argument('--lamb', default=0.05, type=float)         
parser.add_argument('--momentum', default=0, type=float)                                    # select the coefficient for the correction of SAM                                           


parser.add_argument('--method', type=str, default='DFedCata')

args = parser.parse_args()
print(args)

set_global_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device(args.cuda)
else:
    device = torch.device("cpu")

if __name__=='__main__':
    ### Generate IID or Heterogeneous Dataset
    if not args.non_iid:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                                     data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s} {:d} clients".format(args.dataset, 'IID', args.total_client))
    else:
        data_obj = DatasetObject(dataset=args.dataset, n_client=args.total_client, seed=args.seed, unbalanced_sgm=0, rule=args.split_rule,
                                     rule_arg=args.split_coef, data_path=args.data_file)
        print("Initialize the Dataset     --->  {:s} {:s}-{:s} {:d} clients".format(args.dataset, args.split_rule, str(args.split_coef), args.total_client))


    ### Generate Model Function
    model_func = lambda: client_model(args.model)
    print("Initialize the Model Func  --->  {:s} model".format(args.model))
    # Instantiate the class, init_model is the initialized model
    init_model = model_func()
    total_trainable_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
    print("                           --->  {:d} parameters".format(total_trainable_params))
    init_par_list = get_mdl_params(init_model)

    ### Generate Server
    server_func = None
    if args.method == 'DFedCata':
        server_func = DFedCata
    else:
        raise NotImplementedError('not implemented method yet')

    _server = server_func(device=device, model_func=model_func, init_model=init_model, init_par_list=init_par_list,
                          datasets=data_obj, method=args.method, args=args)
    _server.train()
