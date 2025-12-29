"""
Neural network models for federated learning.

This module contains various model architectures used in federated learning experiments,
including CNNs, ResNets, and RNNs for different datasets and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class client_model(nn.Module):
    """
    Unified model class that supports multiple neural network architectures.

    This class provides a flexible interface to instantiate different model architectures
    based on the 'name' parameter. Each model is optimized for specific datasets and
    federated learning scenarios.

    Supported models:
    - LeNet: For MNIST and CIFAR datasets
    - ResNet18: For CIFAR-10/100 and TinyImageNet
    - LSTM variants: For text classification tasks
    - Custom architectures: For specialized tasks

    Args:
        name (str): Model architecture name
        args: Additional arguments (used for some models)
    """

    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name

        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)

        if self.name == 'mnist_2NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)

        if self.name == 'emnist_NN':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)

        if self.name == 'LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'ResNet18':
            resnet18 = models.resnet18(pretrained= True)
            resnet18.fc = nn.Linear(512, 100)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18
        
        if self.name == 'ResNet18_tinyimagenet':
            # Tiny-ImageNet: 64x64 images, 200 classes
            resnet18 = models.resnet18(pretrained=False)
            # Modify first convolution: use 3x3 kernel, stride=1, better for small images
            resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet18.maxpool = nn.Identity()  # Remove maxpool, preserve more spatial information
            resnet18.fc = nn.Linear(512, 200)  # 200 classes

            # Change BN to GN (GroupNorm is more stable in federated learning)
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            self.model = resnet18
            
        if self.name == 'ResNet18_cifar10':
            resnet18 = models.resnet18(pretrained= self.args.pretrained)
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'

            self.model = resnet18

        if self.name == 'ShakespeareLSTM':
            # Shakespeare character-level language model
            # Input: [batch_size, seq_len] character index sequences
            # Output: [batch_size, num_classes] probability distribution of next character
            vocab_size = 80  # ALL_LETTERS character set size
            embedding_dim = 8
            hidden_dim = 256
            num_classes = 80
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        if self.name == 'SentimentLSTM':
            # Sent140 sentiment classification model (Twitter binary classification)
            # Input: [batch_size, seq_len] word index sequences
            # Output: [batch_size, 2] sentiment probability distribution (negative/positive)
            vocab_size = 10000  # vocabulary size
            embedding_dim = 128
            hidden_dim = 256
            num_classes = 2
            
            self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                               batch_first=True, dropout=0.3, bidirectional=True)
            self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)

        if self.name == 'mnist_2NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'emnist_NN':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'ResNet18':
            x = self.model(x)

        if self.name == 'ResNet18_tinyimagenet':
            x = self.model(x)

        if self.name == 'ResNet18_cifar10':
            x = self.model(x)
        
        if self.name == 'ShakespeareLSTM':
            # x shape: [batch_size, seq_len]
            embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim]
            # Use output from last time step
            x = self.fc(lstm_out[:, -1, :])  # [batch_size, num_classes]
        
        if self.name == 'SentimentLSTM':
            # x shape: [batch_size, seq_len]
            embeds = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
            lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim*2]
            # Use output from last time step
            x = F.relu(self.fc1(lstm_out[:, -1, :]))
            x = self.dropout(x)
            x = self.fc2(x)  # [batch_size, 2]
        
        return x

