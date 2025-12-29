import datetime
import os
import time
import numpy as np
import random

import torch
from utils import *
from dataset import Dataset
from torch.utils import data
import networkx as nx

from utils import *


class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        # super(Server, self).__init__()
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func

        self.server_model = init_model
        self.server_model_params_list = init_par_list
        self.max_acc = 0
        self.datetimes = datetime.datetime.now()
        self.total_params = sum(p.numel() for p in init_model.parameters() if p.requires_grad)
        self.t = 0

        print("Initialize the Server      --->  {:s}".format(self.args.method))
        ### Generate Storage
        print("Initialize the Public Storage:")
        # clients_params_list repeats server model parameters for total_client times, size is (args.total_client, number of learnable parameters)
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        self.clients_params_list_old = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
            self.clients_params_list.shape[0], self.clients_params_list.shape[1]))

        self.train_perf = np.zeros((self.args.comm_rounds, 2))
        self.test_perf = np.zeros((self.args.comm_rounds, 2))
        print("   Train/Test [loss, acc]  --->  {:d} * {:d}".format(self.train_perf.shape[0], self.train_perf.shape[1]))
        ### Generate Log Storage : [[E||wi - w||]...] * T
        self.divergence = np.zeros((args.comm_rounds))
        print("  Consistency (Divergence) --->  {:d}".format(self.divergence.shape[0]))

        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate

        # transfer vectors (must be defined if use)
        self.comm_vecs = {
            'Params_list': None,
        }
        self.received_vecs = None
        self.Client = None

    def _see_the_watch_(self):
        # see time
        self.time.append(datetime.datetime.now())

    def _see_the_divergence_(self, selected_clients, t):
        # calculate the divergence
        self.divergence[t] = torch.norm(self.clients_params_list[selected_clients] - \
                                        self.server_model_params_list) ** 2 / len(selected_clients)

    def _activate_clients_(self, t):
        # select active clients ID
        inc_seed = 0
        while (True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients

    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay

    def _test_(self, t, selected_clients):
        # test
        # validation on train set
        loss, acc = self._validate_((np.concatenate(self.datasets.client_x, axis=0), np.concatenate(self.datasets.client_y, axis=0)))
        self.train_perf[t] = [loss, acc]
        print("   Train    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.train_perf[t][0], self.train_perf[t][1]), flush=True)
        # validation on test set
        loss, acc = self._validate_((self.datasets.test_x, self.datasets.test_y))
        self.test_perf[t] = [loss, acc]
        print("    Test    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(
                self.test_perf[t][0], self.test_perf[t][1]), flush=True)
        # calculate consistency
        self._see_the_divergence_(selected_clients, t)
        print("            ----    Divergence: {:.4f}".format(self.divergence[t]), flush=True)

    def _summary_(self):
        # print results summary
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print("     Communication round   --->   T = {:d}       ".format(self.args.comm_rounds))
        print("    Average Time / round   --->   {:.2f}s        ".format(np.mean(self.time)))
        print("     Top-1 Test Acc (T)    --->   {:.2f}% ({:d}) ".format(np.max(self.test_perf[:, 1]) * 100., np.argmax(self.test_perf[:, 1])))
        if self.args.dataset == 'shakespeare':
            best_loss = np.min(self.test_perf[:, 0])
            best_ppl = np.exp(best_loss)
            print("     Best Perplexity (T)   --->   {:.2f} ({:d}) ".format(best_ppl, np.argmin(self.test_perf[:, 0])))

    def _validate_(self, dataset):
        self.server_model.eval()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        testdataset = data.DataLoader(Dataset(dataset[0], dataset[1], train=False, dataset_name=self.args.dataset), batch_size=1000, shuffle=False)

        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                predictions = self.server_model(inputs)
                loss = self.loss(predictions, labels)
                total_loss += loss.item()

                predictions = predictions.cpu().numpy()
                predictions = np.argmax(predictions, axis=1).reshape(-1)
                labels = labels.cpu().numpy().reshape(-1).astype(np.int32)
                batch_correct = np.sum(predictions == labels)
                total_acc += batch_correct

        if self.args.weight_decay != 0.:
            # Add L2 loss
            total_loss += self.args.weight_decay / 2. * torch.sum(self.server_model_params_list * self.server_model_params_list)

        return total_loss / (i + 1), total_acc / dataset[0].shape[0]

    def _save_results_(self):
        # save results.npy
        options = ''  # use '-' at first if non-empty

        root = '{:s}/T={:d}'.format(self.args.out_file, self.args.comm_rounds)
        if not os.path.exists(root):
            os.makedirs(root)
        if not self.args.non_iid:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, 'IID',
                                                 '', self.args.total_client)
        else:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, self.args.split_rule,
                                                 str(self.args.split_coef), self.args.total_client)
        if not os.path.exists(root):
            os.makedirs(root)

        participation = str(self.args.active_ratio)
        root = root + '/active-' + participation

        if not os.path.exists(root):
            os.makedirs(root)

        # save [loss, acc] results
        perf_dir = root + '/Performance'
        if not os.path.exists(perf_dir):
            os.makedirs(perf_dir)
        train_file = perf_dir + '/trn-M_{:s}-rho_{:.3f}-lamb_{:.3f}-momen_{:.3f}-K_{:d}.npy'.format(self.args.method, self.args.rho, self.args.lamb ,self.args.momentum,self.args.local_epochs)
        test_file = perf_dir + '/tst-M_{:s}-rho_{:.3f}-lamb_{:.3f}-momen_{:.3f}-K_{:d}.npy'.format(self.args.method, self.args.rho, self.args.lamb ,self.args.momentum,self.args.local_epochs)
        np.save(train_file, self.train_perf)
        np.save(test_file, self.test_perf)

        # save [divergence, consistency] results
        divergence_dir = root + '/Divergence'
        if not os.path.exists(divergence_dir):
            os.makedirs(divergence_dir)
        divergence_file = divergence_dir + '/divergence-M_{:s}-rho_{:.3f}-lamb_{:.3f}-momen_{:.3f}-K_{:d}.npy'.format(self.args.method, self.args.rho, self.args.lamb ,self.args.momentum,self.args.local_epochs)
        np.save(divergence_file, self.divergence)

    def process_for_communication(self ,client ,nei_indexs):
        pass

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass

    def postprocess(self, client):
        pass

    def neighbor_comm(self, mode ,client):
        pass

    def select_neighbor(self, lst, selected_value, select_ratio, random_seed=None):
        random.seed(random_seed)
        if selected_value == None:
            selected_value = random.choice(lst)
        remaining_values = [value for value in lst if value != selected_value]
        list_length = len(remaining_values)
        select_num = max(1, int(list_length * select_ratio))
        sub_list = [selected_value] * select_num
        selected_values = [random.choice(remaining_values) for _ in range(select_num - 1)]
        sub_list[1:] = selected_values

        return sub_list

    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, cs = False):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes
        if cs == "random":
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + cur_clnt)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])

        elif cs == "grid":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            top = (cur_clnt - 9 + client_num_in_total) % client_num_in_total
            down = (cur_clnt + 9 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([left, right, top, down])

        elif cs =="exp": # (2^6<100<2^7)
            n1 = (cur_clnt + 1 + client_num_in_total) % client_num_in_total
            n2 = (cur_clnt + 2 + client_num_in_total) % client_num_in_total
            n3 = (cur_clnt + 4 + client_num_in_total) % client_num_in_total
            n4 = (cur_clnt + 8 + client_num_in_total) % client_num_in_total
            n5 = (cur_clnt + 16 + client_num_in_total) % client_num_in_total
            n6 = (cur_clnt + 32 + client_num_in_total) % client_num_in_total
            n7 = (cur_clnt + 64 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([n1,n2,n3,n4,n5,n6,n7])

        elif cs == "full":
            client_indexes = np.arange(client_num_in_total)
            client_indexes = np.delete(client_indexes, cur_clnt)

        return client_indexes

    def generate_random_topo(self, client_num_in_total, client_num_per_round,t ,seed):
        A = np.zeros((client_num_in_total, client_num_in_total))
        np.random.seed(seed + t)
        for i in range(client_num_in_total):
            indices = np.random.choice(client_num_in_total, client_num_per_round - 1, replace=False)
            A[i, indices] = 1.0
            A[i, i] = 0
        B = np.zeros((client_num_in_total, client_num_in_total))
        for i in range(client_num_in_total):
            for j in range(i, client_num_in_total):
                B[i, j] = A[i, j]
        B = B + B.T - np.diag(B.diagonal())
        return B

    def generate_er_topo(self, client_num_in_total, t, seed):

        np.random.seed(seed + t)
        if self.args.cs == "er":
            G = nx.erdos_renyi_graph(client_num_in_total, self.args.active_ratio)
        elif self.args.cs == "ws":
            G = nx.watts_strogatz_graph(client_num_in_total, 8, 0.1, seed=seed)
        adj_matrix = nx.adjacency_matrix(G)

        dense_adj_matrix = adj_matrix.todense()

        return dense_adj_matrix

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        for t in range(self.args.comm_rounds):
            self.t = t
            start = time.time()

            print('============= Communication Round', t + 1, '=============', flush=True)
            if self.args.cs == "random":
                topo = self.generate_random_topo(self.args.total_client,
                                             int(self.args.total_client * self.args.active_ratio),t,self.args.seed)
            elif self.args.cs == "er" or self.args.cs == "ws":
                topo = self.generate_er_topo(self.args.total_client, t, self.args.seed)
            for client in range(self.args.total_client):
                if self.args.cs == "random":
                    nei_indexs = np.nonzero(topo[client])[0]
                    if self.args.total_client != int(self.args.total_client * self.args.active_ratio):
                        if client not in nei_indexs:
                            nei_indexs = np.append(nei_indexs, client)
                    nei_indexs = np.sort(nei_indexs)
                elif self.args.cs == "er" or self.args.cs == "ws":
                    nei_indexs = np.nonzero(topo[client])[1]
                    if self.args.total_client != int(self.args.total_client * self.args.active_ratio):
                        if client not in nei_indexs:
                            nei_indexs = np.append(nei_indexs, client)
                    nei_indexs = np.sort(nei_indexs)
                else:
                    nei_indexs = self._benefit_choose(t, client, self.args.total_client,
                                                  int(self.args.total_client * self.args.active_ratio), self.args.cs)
                    if self.args.total_client != int(self.args.total_client * self.args.active_ratio):
                        nei_indexs = np.append(nei_indexs, client)
                    nei_indexs = np.sort(nei_indexs)

                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client , nei_indexs)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                           dataset=dataset, lr=self.lr, args=self.args)
                self.received_vecs = _edge_device.train()
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client) 
                del _edge_device

            self.clients_params_list_old = self.clients_params_list


            Averaged_model = torch.mean(self.clients_params_list, dim=0)
            set_client_from_params(self.device, self.server_model, Averaged_model)
            self._test_(t, selected_clients = list(range(self.args.total_client)))
            self._lr_scheduler_()
            end = time.time()
            self.time[t] = end - start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush=True)

        self._summary_()
        self._save_results_()

