import os
import numpy as np
import scipy.io as io
from PIL import Image
import warnings

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
import requests
import zipfile
from tqdm import tqdm

# Disable SSL warnings
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        # self.name = "{:s}_{:s}_{:s}_{:.0f}%-{:d}".format(dataset, rule, str(rule_arg), args.active_ratio*args.total_client, args.total_client)
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            # Automatically download and process Tiny-ImageNet
            if self.dataset == 'tinyimagenet':
                raw_path = f"{self.data_path}Data/Raw/tiny-imagenet-200"
                if not os.path.exists(raw_path):
                    print("Downloading Tiny-ImageNet dataset...")
                    self._download_tinyimagenet()
                self._process_tinyimagenet(raw_path)
            
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trainset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trainset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trainset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            
            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([# transforms.Resize(224),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], #pre-train
                                                #                      std=[0.229, 0.224, 0.225])])
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                     std=[0.5, 0.5, 0.5])])
                # trainset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=True , download=True, transform=transform)
                # testset = torchvision.datasets.ImageFolder(root='%sData/Raw' %self.data_path,
                #                                       train=False, download=True, transform=transform)
                # root_dir = self.data_path
                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list, transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list, transformer=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            if self.dataset != 'emnist':
                train_itr = train_load.__iter__(); test_itr = test_load.__iter__() 
                # labels are of shape (n_data,)
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
                test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                train_idx = np.where(y_train < 10)[0]

                y_train = y_train[train_idx]
                x_train = x_train[train_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                test_idx = np.where(y_test < 10)[0]

                y_test = y_test[test_idx]
                x_test = x_test[test_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                train_x = (x_train - mean_x) / std_x
                train_y = y_train
                
                test_x = (x_test  - mean_x) / std_x
                test_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]
            
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
            
            
            ###
            n_data_per_client = int((len(train_y)) / self.n_client)
            # Draw from lognormal distribution
            # client_data_list = (np.random.lognormal(mean=np.log(n_data_per_client), sigma=self.unbalanced_sgm, size=self.n_client))
            # client_data_list = (client_data_list/(np.sum(client_data_list)/len(train_y)))
            client_data_list = np.ones(self.n_client, dtype=int)*n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break
            ###     
            
            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client,self.n_cls])
                    a[:,c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i,j] != 0:
                                prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0

                idx_list = [np.where(train_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
    
                while(np.sum(client_data_list)!=0):
                    curr_client = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount [cls_label] = len(idx_list[cls_label]) 
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)
                
                # cls_means = np.zeros((self.n_client, self.n_cls))
                # for client in range(self.n_client):
                #     for cls in range(self.n_cls):
                #         cls_means[client,cls] = np.mean(client_y[client]==cls)
                # prior_real_diff = np.abs(cls_means-cls_priors)
                # print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                # print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(train_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(train_y[:, 0])
                n_data_per_client = len(train_y) // self.n_client
                # client_x dtype needs to be float32, the same as weights
                client_x = np.zeros((self.n_client, n_data_per_client, 3, 32, 32), dtype=np.float32)
                client_y = np.zeros((self.n_client, n_data_per_client, 1), dtype=np.float32)
                train_x = train_x[idx] # 50000*3*32*32
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_client // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        client_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        client_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
            
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)

            
            self.client_x = client_x; self.client_y = client_y

            self.test_x  = test_x;  self.test_y  = test_y
            
            # Save data
            print('begin to save data...')
            os.mkdir('%sData/%s' %(self.data_path, self.name))
            
            np.save('%sData/%s/client_x.npy' %(self.data_path, self.name), client_x)
            np.save('%sData/%s/client_y.npy' %(self.data_path, self.name), client_y)

            np.save('%sData/%s/test_x.npy'  %(self.data_path, self.name),  test_x)
            np.save('%sData/%s/test_y.npy'  %(self.data_path, self.name),  test_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            self.client_x = np.load('%sData/%s/client_x.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.client_y = np.load('%sData/%s/client_y.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.n_client = len(self.client_x)

            self.test_x  = np.load('%sData/%s/test_x.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            self.test_y  = np.load('%sData/%s/test_y.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'tinyimagenet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            print('data loading finished.')
                
        '''
        print('Class frequencies:')
        count = 0
        for client in range(self.n_client):
            print("Client %3d: " %client + 
                  ', '.join(["%.3f" %np.mean(self.client_y[client]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.client_y[client].shape[0])
            count += self.client_y[client].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.test_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.test_y.shape[0])
        '''
    
    def _download_tinyimagenet(self):
        """Download and extract Tiny-ImageNet dataset"""
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        save_dir = f"{self.data_path}Data/Raw"
        zip_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
        
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Download dataset (disable proxy to avoid proxy issues)
            print(f"Downloading from {url}...")
            print("If download fails, please manually download and extract to:", os.path.join(save_dir, "tiny-imagenet-200"))
            
            # Disable system proxy
            session = requests.Session()
            session.trust_env = False
            session.proxies = {'http': None, 'https': None}
            
            response = session.get(url, stream=True, verify=False, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
            
            # Extract dataset
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            
            # Clean up ZIP file
            print("Cleaning up...")
            os.remove(zip_path)
            print("Download and extraction complete!")
            
        except Exception as e:
            print(f"\n❌ Automatic download failed: {str(e)}")
            print("\nPlease manually download Tiny-ImageNet dataset:")
            print(f"1. Download URL: {url}")
            print(f"   Or mirror URL: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print(f"2. Extract to: {os.path.abspath(save_dir)}")
            print(f"3. Ensure directory structure: {os.path.abspath(save_dir)}/tiny-imagenet-200/train, val, test")
            print("\nThen run the program again.")
            raise RuntimeError("Automatic download failed, please manually download as instructed above.")

    def _process_tinyimagenet(self, root_dir):
        """Generate train_list.txt and val_list.txt for Tiny-ImageNet"""
        train_list_path = os.path.join(root_dir, 'train_list.txt')
        val_list_path = os.path.join(root_dir, 'val_list.txt')
        
        if os.path.exists(train_list_path) and os.path.exists(val_list_path):
            print("train_list.txt and val_list.txt already exist, skipping processing.")
            return
        
        print("Processing Tiny-ImageNet dataset...")
        
        # Process training set
        train_dir = os.path.join(root_dir, 'train')
        train_list = []
        class_to_idx = {}
        
        # Create class name to index mapping (sorted alphabetically for consistency)
        for idx, class_name in enumerate(sorted(os.listdir(train_dir))):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                class_to_idx[class_name] = idx
        
        # Generate training set list
        for class_name in sorted(os.listdir(train_dir)):
            class_dir = os.path.join(train_dir, class_name, 'images')
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith('.JPEG'):
                    rel_path = os.path.join('train', class_name, 'images', img_name)
                    train_list.append(f"{rel_path} {class_to_idx[class_name]}")
        
        print(f"Writing {len(train_list)} training samples to {train_list_path}")
        with open(train_list_path, 'w') as f:
            f.write("\n".join(train_list))
        
        # Process validation set
        val_dir = os.path.join(root_dir, 'val')
        annotations = {}
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        
        with open(val_annotations_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_name = parts[0], parts[1]
                    annotations[img_name] = class_name
        
        val_list = []
        val_images_dir = os.path.join(val_dir, 'images')
        for img_name in sorted(os.listdir(val_images_dir)):
            if img_name.endswith('.JPEG') and img_name in annotations:
                class_name = annotations[img_name]
                if class_name in class_to_idx:
                    rel_path = os.path.join('val', 'images', img_name)
                    val_list.append(f"{rel_path} {class_to_idx[class_name]}")
        
        print(f"Writing {len(val_list)} validation samples to {val_list_path}")
        with open(val_list_path, 'w') as f:
            f.write("\n".join(val_list))
        
        print("Tiny-ImageNet processing complete!")
        
def generate_syn_logistic(dimension, n_client, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_client)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_client))
    data_y = list(range(n_client))

    mean_W = np.random.normal(0, alpha, n_client)
    B = np.random.normal(0, beta, n_client)

    mean_x = np.zeros((n_client, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_client):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_client):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y

    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet":
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
        
        elif self.name == 'shakespeare':
            # Shakespeare data needs to be int64 type
            self.X_data = torch.from_numpy(np.asarray(data_x, dtype=np.int64)).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.from_numpy(np.asarray(data_y, dtype=np.int64)).long()
        
        elif self.name == 'sent140':
            # Sent140 sentiment classification data
            self.X_data = torch.from_numpy(np.asarray(data_x, dtype=np.int64)).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.from_numpy(np.asarray(data_y, dtype=np.int64)).long()
                
        else:
            raise NotImplementedError
            
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
                
        elif self.name == 'tinyimagenet':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if np.random.rand() > .5:
                    # Random cropping
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
        
        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            if isinstance(self.y_data, bool):
                return x
            else:
                y = self.y_data[idx]
                return x, y
        
        elif self.name == 'sent140':
            x = self.X_data[idx]
            if isinstance(self.y_data, bool):
                return x
            else:
                y = self.y_data[idx]
                return x, y

        else:
            raise NotImplementedError
            
class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list) 


# ============= Shakespeare Dataset Support =============

class ShakespeareObjectCrop_h5:
    """Shakespeare dataset (loaded from h5 file, naturally non-iid partitioned)"""
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, test_ratio=5, rand_seed=0, n_client=None):
        import h5py
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        
        train_h5 = os.path.join(data_path, 'shakespeare_train.h5')
        test_h5 = os.path.join(data_path, 'shakespeare_test.h5')
        
        if not os.path.exists(train_h5) or not os.path.exists(test_h5):
            raise FileNotFoundError(
                f"H5 data file not found. Please ensure the following files exist:\n"
                f"  {train_h5}\n  {test_h5}\n"
                f"You can run python download_real_data.py to download data, then extract shakespeare.tar.bz2"
            )
        
        f_train = h5py.File(train_h5, 'r')
        f_test = h5py.File(test_h5, 'r')
        
        all_users = list(f_train['examples'].keys())
        
        # First filter clients with valid data
        valid_users_data = []
        for user in all_users:
            snippets = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in f_train['examples'][user]['snippets'][:]]
            # Check if there are sufficiently long snippets
            has_data = any(len(s) > 80 for s in snippets)
            if has_data:
                valid_users_data.append(user)
        
        # Limit client count (select from valid clients)
        if n_client is not None and n_client < len(valid_users_data):
            np.random.seed(rand_seed)
            selected_indices = np.random.choice(len(valid_users_data), n_client, replace=False)
            users = [valid_users_data[i] for i in sorted(selected_indices)]
        else:
            users = valid_users_data
        
        self.client_x = []
        self.client_y = []
        
        # 处理训练数据
        for client, user in enumerate(users):
            np.random.seed(rand_seed + client)
            snippets = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in f_train['examples'][user]['snippets'][:]]
            
            # 将 snippets 转换为 (x, y) 对
            x_list, y_list = [], []
            for snippet in snippets:
                for i in range(0, len(snippet) - 80):
                    x_list.append(snippet[i:i+80])
                    y_list.append(snippet[i+80])
            
            # 裁剪数据量
            if len(x_list) > crop_amount:
                start = np.random.randint(len(x_list) - crop_amount)
                x_list = x_list[start:start + crop_amount]
                y_list = y_list[start:start + crop_amount]
            
            self.client_x.append(np.array(x_list, dtype=object))
            self.client_y.append(np.array(y_list, dtype=object))
        
        self.users = users
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        
        # 处理测试数据
        test_x_list, test_y_list = [], []
        test_users = list(f_test['examples'].keys())
        for user in test_users:
            snippets = [s.decode('utf-8') if isinstance(s, bytes) else s 
                        for s in f_test['examples'][user]['snippets'][:]]
            for snippet in snippets:
                for i in range(0, len(snippet) - 80):
                    test_x_list.append(snippet[i:i+80])
                    test_y_list.append(snippet[i+80])
        
        # 限制测试集大小
        max_test = (crop_amount // test_ratio) * self.n_client
        if len(test_x_list) > max_test:
            np.random.seed(rand_seed)
            indices = np.random.choice(len(test_x_list), max_test, replace=False)
            test_x_list = [test_x_list[i] for i in indices]
            test_y_list = [test_y_list[i] for i in indices]
        
        self.test_x = np.array(test_x_list, dtype=object)
        self.test_y = np.array(test_y_list, dtype=object)
        
        f_train.close()
        f_test.close()
        
        # 转换字符为索引
        self._convert_to_indices()
        
        self.channels = 1
        self.width = 80
        self.height = 1
        self.n_cls = NUM_LETTERS
        
        print(f"Shakespeare h5 data loading completed: {self.n_client} clients, "
              f"test set {len(self.test_x)} samples")
    
    def _convert_to_indices(self):
        """Convert characters to indices"""
        for client in range(len(self.client_x)):
            client_list_x = []
            client_list_y = []
            for idx in range(len(self.client_x[client])):
                client_list_x.append(word_to_indices(self.client_x[client][idx]))
                client_list_y.append(ALL_LETTERS.index(self.client_y[client][idx]) 
                                     if self.client_y[client][idx] in ALL_LETTERS else 0)
            # 使用 int64 类型，便于后续转为 tensor
            self.client_x[client] = np.array(client_list_x, dtype=np.int64)
            self.client_y[client] = np.array(client_list_y, dtype=np.int64).reshape(-1, 1)
        
        test_list_x = []
        test_list_y = []
        for idx in range(len(self.test_x)):
            test_list_x.append(word_to_indices(self.test_x[idx]))
            test_list_y.append(ALL_LETTERS.index(self.test_y[idx]) 
                               if self.test_y[idx] in ALL_LETTERS else 0)
        self.test_x = np.array(test_list_x, dtype=np.int64)
        self.test_y = np.array(test_list_y, dtype=np.int64).reshape(-1, 1)
        
        # 转为 numpy 数组
        self.client_x = np.asarray(self.client_x, dtype=object)
        self.client_y = np.asarray(self.client_y, dtype=object)


# ============= Sentiment140 Dataset Support =============

class Sent140Dataset:
    """Sentiment140 sentiment classification dataset (Twitter, naturally non-iid)"""
    def __init__(self, data_path='./', n_client=100, max_seq_len=25, max_vocab=10000, 
                 crop_amount=500, test_ratio=5, rand_seed=0):
        self.dataset = 'sent140'
        self.name = f'sent140_{n_client}_{rand_seed}'
        self.max_seq_len = max_seq_len
        self.max_vocab = max_vocab
        
        h5_train = os.path.join(data_path, 'sent140_train.h5')
        h5_test = os.path.join(data_path, 'sent140_test.h5')
        
            # If data doesn't exist, download automatically
        if not os.path.exists(h5_train) or not os.path.exists(h5_test):
            self._download_sent140(data_path)
        
        if os.path.exists(h5_train) and os.path.exists(h5_test):
            self._load_from_h5(h5_train, h5_test, n_client, crop_amount, test_ratio, rand_seed)
        else:
            # If download fails, use synthetic data for demonstration
            print("⚠ Unable to download Sent140 data, using synthetic sentiment data for demonstration...")
            self._generate_synthetic(n_client, crop_amount, test_ratio, rand_seed)
        
        self.channels = 1
        self.width = max_seq_len
        self.height = 1
        self.n_cls = 2  # Binary classification: positive/negative
        
        print(f"Sent140 data loading completed: {self.n_client} clients, test set {len(self.test_x)} samples")
    
    def _download_sent140(self, data_path):
        """Download and process Sent140 dataset using LEAF format"""
        import csv
        from collections import defaultdict
        import h5py
        
        leaf_data_dir = os.path.join(data_path, 'leaf', 'data', 'sent140', 'data')
        raw_data_dir = os.path.join(leaf_data_dir, 'raw_data')
        
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # 1. Download raw data
        zip_path = os.path.join(raw_data_dir, 'trainingandtestdata.zip')
        train_csv = os.path.join(raw_data_dir, 'training.csv')
        test_csv = os.path.join(raw_data_dir, 'test.csv')
        
        if not os.path.exists(train_csv):
            print("Downloading Sentiment140 dataset from Stanford...")
            url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
            
            try:
                # 下载 ZIP 文件
                print(f"Downloading: {url}")
                response = requests.get(url, stream=True, verify=False, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载进度") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # 解压
                print("解压数据...")
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_data_dir)
                
                # 重命名文件
                orig_train = os.path.join(raw_data_dir, 'training.1600000.processed.noemoticon.csv')
                orig_test = os.path.join(raw_data_dir, 'testdata.manual.2009.06.14.csv')
                if os.path.exists(orig_train):
                    os.rename(orig_train, train_csv)
                if os.path.exists(orig_test):
                    os.rename(orig_test, test_csv)
                
                # 清理 ZIP
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    
                print("✓ 下载完成!")
                
            except Exception as e:
                print(f"下载失败: {e}")
                return False
        
        # 2. 处理数据为 h5 格式
        print("处理数据为联邦学习格式...")
        return self._process_leaf_sent140(train_csv, test_csv, data_path)
    
    def _process_leaf_sent140(self, train_csv, test_csv, data_path):
        """处理 LEAF 格式的 Sent140 数据"""
        import csv
        import h5py
        from collections import defaultdict
        
        # 构建词汇表
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        word_freq = defaultdict(int)
        
        # 读取训练数据并统计词频
        print("读取训练数据...")
        train_data = []
        with open(train_csv, 'rt', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, desc="读取CSV"):
                # CSV 格式: sentiment, id, date, query, user, text
                if len(row) >= 6:
                    sentiment = int(row[0])  # 0=负面, 4=正面
                    user = row[4]
                    text = row[5]
                    train_data.append((sentiment, user, text))
                    for word in text.lower().split():
                        word_freq[word] += 1
        
        print(f"训练样本数: {len(train_data)}")
        
        # 构建词汇表
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab - 2]:
            word_to_idx[word] = len(word_to_idx)
        
        def text_to_indices(text):
            words = text.lower().split()[:self.max_seq_len]
            indices = [word_to_idx.get(w, 1) for w in words]
            if len(indices) < self.max_seq_len:
                indices = indices + [0] * (self.max_seq_len - len(indices))
            return indices
        
        # 按用户分组（天然 non-iid）
        user_data = defaultdict(lambda: {'x': [], 'y': []})
        for sentiment, user, text in tqdm(train_data, desc="按用户分组"):
            label = 1 if sentiment == 4 else 0
            indices = text_to_indices(text)
            user_data[user]['x'].append(indices)
            user_data[user]['y'].append(label)
        
        # 过滤掉样本太少的用户
        valid_users = {u: d for u, d in user_data.items() if len(d['x']) >= 5}
        print(f"有效用户数: {len(valid_users)} (至少5个样本)")
        
        # 保存训练数据
        train_h5_path = os.path.join(data_path, 'sent140_train.h5')
        with h5py.File(train_h5_path, 'w') as f:
            examples = f.create_group('examples')
            for user_id, data in tqdm(valid_users.items(), desc="保存训练数据"):
                user_group = examples.create_group(user_id)
                user_group.create_dataset('x', data=np.array(data['x'], dtype=np.int64))
                user_group.create_dataset('y', data=np.array(data['y'], dtype=np.int64))
        
        # 处理测试数据
        print("处理测试数据...")
        test_x, test_y = [], []
        with open(test_csv, 'rt', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    sentiment = int(row[0])
                    text = row[5]
                    label = 1 if sentiment == 4 else 0
                    indices = text_to_indices(text)
                    test_x.append(indices)
                    test_y.append(label)
        
        # 保存测试数据
        test_h5_path = os.path.join(data_path, 'sent140_test.h5')
        with h5py.File(test_h5_path, 'w') as f:
            examples = f.create_group('examples')
            test_group = examples.create_group('test')
            test_group.create_dataset('x', data=np.array(test_x, dtype=np.int64))
            test_group.create_dataset('y', data=np.array(test_y, dtype=np.int64))
        
        print(f"✓ 数据处理完成!")
        print(f"  训练用户数: {len(valid_users)}")
        print(f"  测试样本数: {len(test_x)}")
        return True
    
    def _load_from_h5(self, train_h5, test_h5, n_client, crop_amount, test_ratio, rand_seed):
        """从 h5 文件加载数据"""
        import h5py
        
        f_train = h5py.File(train_h5, 'r')
        f_test = h5py.File(test_h5, 'r')
        
        all_users = list(f_train['examples'].keys())
        
        # 限制客户端数量
        if n_client < len(all_users):
            np.random.seed(rand_seed)
            selected_indices = np.random.choice(len(all_users), n_client, replace=False)
            users = [all_users[i] for i in sorted(selected_indices)]
        else:
            users = all_users[:n_client]
        
        self.users = users
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        
        self.client_x = []
        self.client_y = []
        
        # 处理训练数据
        for client, user in enumerate(users):
            np.random.seed(rand_seed + client)
            
            x_data = f_train['examples'][user]['x'][:]
            y_data = f_train['examples'][user]['y'][:]
            
            # 裁剪数据量
            if len(x_data) > crop_amount:
                indices = np.random.choice(len(x_data), crop_amount, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
            
            self.client_x.append(np.asarray(x_data, dtype=np.int64))
            self.client_y.append(np.asarray(y_data, dtype=np.int64).reshape(-1, 1))
        
        # 处理测试数据（测试数据统一存储在 'test' 键下）
        if 'test' in f_test['examples']:
            # 新格式：所有测试数据在一个 'test' 组中
            test_x_list = f_test['examples']['test']['x'][:]
            test_y_list = f_test['examples']['test']['y'][:]
        else:
            # 旧格式：按用户分组
            test_x_list, test_y_list = [], []
            test_users = list(f_test['examples'].keys())
            for user in test_users:
                x_data = f_test['examples'][user]['x'][:]
                y_data = f_test['examples'][user]['y'][:]
                test_x_list.extend(x_data)
                test_y_list.extend(y_data)
            test_x_list = np.array(test_x_list)
            test_y_list = np.array(test_y_list)
        
        # 限制测试集大小
        max_test = (crop_amount // test_ratio) * self.n_client
        if len(test_x_list) > max_test:
            np.random.seed(rand_seed)
            indices = np.random.choice(len(test_x_list), max_test, replace=False)
            test_x_list = test_x_list[indices]
            test_y_list = test_y_list[indices]
        
        self.test_x = np.asarray(test_x_list, dtype=np.int64)
        self.test_y = np.asarray(test_y_list, dtype=np.int64).reshape(-1, 1)
        
        f_train.close()
        f_test.close()
        
        self.client_x = np.asarray(self.client_x, dtype=object)
        self.client_y = np.asarray(self.client_y, dtype=object)
    
    def _generate_synthetic(self, n_client, crop_amount, test_ratio, rand_seed):
        """生成合成情感数据用于测试"""
        np.random.seed(rand_seed)
        
        self.n_client = n_client
        self.user_idx = np.asarray(list(range(self.n_client)))
        
        # 为每个客户端生成数据（模拟 non-iid：不同客户端有不同的情感偏向）
        self.client_x = []
        self.client_y = []
        
        for client in range(n_client):
            np.random.seed(rand_seed + client)
            # 每个客户端有不同的正负样本比例（non-iid）
            pos_ratio = np.random.beta(2, 2)  # 0.2 到 0.8 之间
            n_samples = min(crop_amount, np.random.randint(100, 500))
            
            # 生成标签
            labels = (np.random.random(n_samples) < pos_ratio).astype(np.int64)
            # 生成随机词索引序列
            sequences = np.random.randint(1, self.max_vocab, (n_samples, self.max_seq_len))
            
            self.client_x.append(sequences)
            self.client_y.append(labels.reshape(-1, 1))
        
        # 生成测试数据
        n_test = (crop_amount // test_ratio) * n_client
        self.test_x = np.random.randint(1, self.max_vocab, (n_test, self.max_seq_len)).astype(np.int64)
        self.test_y = np.random.randint(0, 2, (n_test, 1)).astype(np.int64)
        
        self.client_x = np.asarray(self.client_x, dtype=object)
        self.client_y = np.asarray(self.client_y, dtype=object)


def read_data(train_data_dir, test_data_dir):
    """读取莎士比亚数据集"""
    import json
    
    def read_dir(data_dir):
        users = []
        groups = []
        data = {}
        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            users.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])
        return users, groups, data
    
    train_users, train_groups, train_data = read_dir(train_data_dir)
    test_users, test_groups, test_data = read_dir(test_data_dir)
    
    return train_users, train_groups, train_data, test_data

# 字符映射表
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def word_to_indices(word):
    """将单词转换为索引列表"""
    indices = []
    for c in word:
        if c in ALL_LETTERS:
            indices.append(ALL_LETTERS.index(c))
        else:
            indices.append(0)  # 未知字符映射到第一个字符
    return indices

def letter_to_vec(letter):
    """将字母转换为one-hot向量"""
    index = ALL_LETTERS.index(letter) if letter in ALL_LETTERS else 0
    vec = np.zeros(NUM_LETTERS)
    vec[index] = 1
    return vec

class ShakespeareObjectCrop:
    """莎士比亚数据集（所有客户端）"""
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, test_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')
        
        self.users = users
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.client_x = list(range(self.n_client))
        self.client_y = list(range(self.n_client))
        
        # 处理训练数据
        for client in range(self.n_client):
            np.random.seed(rand_seed + client)
            if len(train_data[users[client]]['x']) > crop_amount:
                start = np.random.randint(len(train_data[users[client]]['x'])-crop_amount)
                self.client_x[client] = np.asarray(train_data[users[client]]['x'])[start:start+crop_amount]
                self.client_y[client] = np.asarray(train_data[users[client]]['y'])[start:start+crop_amount]
            else:
                self.client_x[client] = np.asarray(train_data[users[client]]['x'])
                self.client_y[client] = np.asarray(train_data[users[client]]['y'])
        
        # 处理测试数据
        test_data_count = (crop_amount//test_ratio) * self.n_client
        self.test_x = list(range(test_data_count))
        self.test_y = list(range(test_data_count))
        
        test_data_count = 0
        for client in range(self.n_client):
            curr_amount = min(crop_amount//test_ratio, len(test_data[users[client]]['x']))
            if curr_amount > 0:
                np.random.seed(rand_seed + client)
                start = 0 if len(test_data[users[client]]['x']) <= curr_amount else \
                        np.random.randint(len(test_data[users[client]]['x'])-curr_amount)
                self.test_x[test_data_count: test_data_count+ curr_amount] = \
                    np.asarray(test_data[users[client]]['x'])[start:start+curr_amount]
                self.test_y[test_data_count: test_data_count+ curr_amount] = \
                    np.asarray(test_data[users[client]]['y'])[start:start+curr_amount]
                test_data_count += curr_amount
        
        self.client_x = np.asarray(self.client_x, dtype=object)
        self.client_y = np.asarray(self.client_y, dtype=object)
        
        # 转换字符为索引
        self._convert_to_indices()
        
        self.channels = 1
        self.width = 80
        self.height = 1
        self.n_cls = NUM_LETTERS
    
    def _convert_to_indices(self):
        """Convert characters to indices"""
        # 转换客户端数据
        for client in range(len(self.client_x)):
            client_list_x = []
            client_list_y = []
            for idx in range(len(self.client_x[client])):
                client_list_x.append(np.asarray(word_to_indices(self.client_x[client][idx])))
                client_list_y.append(np.argmax(np.asarray(letter_to_vec(self.client_y[client][idx]))))
            self.client_x[client] = np.asarray(client_list_x, dtype=object)
            self.client_y[client] = np.asarray(client_list_y).reshape(-1, 1)
        
        # 转换测试数据
        test_list_x = []
        test_list_y = []
        for idx in range(len(self.test_x)):
            test_list_x.append(np.asarray(word_to_indices(self.test_x[idx])))
            test_list_y.append(np.argmax(np.asarray(letter_to_vec(self.test_y[idx]))))
        self.test_x = np.asarray(test_list_x, dtype=object)
        self.test_y = np.asarray(test_list_y).reshape(-1, 1) 