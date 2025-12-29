import torch
from client import *
from .server import Server


class DFedCata(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(DFedCata, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)

        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        self.Client = dfedcata
    
    
    def process_for_communication(self, client , nei_index):
        self.comm_vecs['Params_list'].copy_(torch.mean(self.clients_params_list[nei_index], dim=0) + self.args.beta\
                                    * (torch.mean(self.clients_params_list[nei_index], dim=0) - self.clients_params_list_old[client]))


    def postprocess(self, client):
        pass
