# DFedCata: Federated Learning Framework

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DFedCata is an efficient federated learning framework that implements the DFedCata algorithm for distributed machine learning scenarios. The framework supports multiple datasets and model architectures, enabling collaborative training while preserving data privacy.

## Features

- 🚀 **Efficient Algorithm**: Implements the DFedCata algorithm to optimize federated learning convergence speed
- 📊 **Multiple Datasets**: Supports MNIST, CIFAR-10, CIFAR-100, and TinyImageNet
- 🏗️ **Flexible Models**: Supports LeNet, ResNet18, and other neural network architectures
- ⚖️ **Data Distribution**: Supports both IID and non-IID data distributions (Dirichlet and Pathological partitioning)
- 🔧 **Easy Extension**: Modular design for easily adding new algorithms and models

## Installation

### Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA (optional, for GPU acceleration)

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/your-username/DFedCata.git
cd DFedCata
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run a simple federated learning experiment:

```bash
python train_1.py --dataset CIFAR10 --model LeNet --total-client 100 --comm-rounds 500
```

### Parameter Description

#### Dataset Parameters
- `--dataset`: Select dataset (`CIFAR10`, `CIFAR100`, `mnist`, `tinyimagenet`)
- `--non-iid`: Enable non-IID data distribution
- `--split-rule`: Data splitting rule (`Dirichlet`, `Path`)
- `--split-coef`: Splitting coefficient (Dirichlet: 0.1-1.0, Path: integer)

#### Model Parameters
- `--model`: Select model architecture (`ResNet18`, `ResNet18_tinyimagenet`, `LeNet`)

#### Training Parameters
- `--total-client`: Total number of clients (default: 100)
- `--active-ratio`: Active client ratio (default: 0.1)
- `--comm-rounds`: Number of communication rounds (default: 500)
- `--local-epochs`: Number of local training epochs (default: 5)
- `--batchsize`: Batch size (default: 128)
- `--local-learning-rate`: Local learning rate (default: 0.1)

#### Algorithm Parameters
- `--method`: Federated learning method (default: DFedCata)
- `--mu`: Client momentum coefficient (default: 0.9)
- `--beta`: Relaxed initialization coefficient (default: 0.9)
- `--rho`: SAM perturbation rate (default: 0)
- `--lamb`: Regularization coefficient (default: 0.05)

### Example Commands

#### CIFAR-10 Experiment (Heterogeneous Data)
```bash
python train_1.py \
    --dataset CIFAR10 \
    --model LeNet \
    --non-iid \
    --split-rule Dirichlet \
    --split-coef 0.3 \
    --total-client 100 \
    --active-ratio 0.1 \
    --comm-rounds 500 \
    --local-epochs 5 \
    --batchsize 128 \
    --local-learning-rate 0.1 \
    --seed 20 \
    --cuda 0
```

#### CIFAR-100 Experiment (Heterogeneous Data Dirichlet)
```bash
python train_1.py \
    --dataset CIFAR100 \
    --model ResNet18 \
    --non-iid \
    --split-rule Dirichlet \
    --split-coef 0.3 \
    --total-client 100 \
    --active-ratio 0.1 \
    --comm-rounds 500 \
    --local-epochs 5 \
    --batchsize 128 \
    --local-learning-rate 0.1 \
    --seed 20 \
    --cuda 0
```

#### CIFAR-100 Experiment (Heterogeneous Data pathological)
```bash
python train_1.py \
    --dataset tinyimagenet \
    --model ResNet18_tinyimagenet \
    --non-iid \
    --split-rule Path \
    --split-coef 20 \
    --total-client 100 \
    --active-ratio 0.1 \
    --comm-rounds 500 \
    --local-epochs 5 \
    --batchsize 128 \
    --local-learning-rate 0.1 \
    --seed 20 \
    --cuda 0
```

## Project Structure

```
DFedCata/
├── client/                 # Client implementations
│   ├── client.py          # Base client class
│   └── dfedcata.py        # DFedCata client implementation
├── server/                 # Server implementations
│   ├── server.py          # Base server class
│   └── DFedCata.py        # DFedCata server implementation
├── optimizer/              # Optimizers
│   └── ESAM.py            # ESAM optimizer
├── dataset.py              # Dataset processing and partitioning
├── models.py               # Model definitions
├── utils.py                # Utility functions
├── train_1.py              # Main training script
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## Supported Algorithms

- **DFedCata**: deventralized federated learning algorithm via Catalyst Framework

## Supported Models

- **LeNet**: Suitable for MNIST dataset
- **ResNet18**: Suitable for CIFAR-10/100 and TinyImageNet
- **ResNet18_tinyimagenet**: ResNet18 optimized for TinyImageNet
- **Others**: mnist_2NN, ShakespeareLSTM, SentimentLSTM, etc.

## Supported Datasets

- **MNIST**: Handwritten digit recognition
- **CIFAR-10/100**: Image classification
- **TinyImageNet**: Small ImageNet subset

## Data Distribution

### IID Distribution
All clients have the same data distribution.

### Non-IID Distribution
- **Dirichlet Distribution**: Uses Dirichlet distribution to simulate heterogeneous data
- **Pathological Distribution**: Each client contains only partial categories of data

## Experimental Results

The project automatically logs experimental results to the `out/` directory, including:
- Training loss and accuracy
- Communication overhead
- Convergence curves

## Extension Development

### Adding New Algorithms

1. Implement new server class in the `server/` directory
2. Implement corresponding client class in the `client/` directory
3. Add algorithm selection logic in `train_1.py`

### Adding New Models

Add new model architectures in the `client_model` class in `models.py`.

### Adding New Datasets

Add new dataset processing logic in the `DatasetObject.set_data()` method in `dataset.py`.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{li2024boosting,
  title={Boosting the Performance of Decentralized Federated Learning via Catalyst Acceleration},
  author={Li, Qinglun and Zhang, Miao and Liu, Yingqi and Yin, Quanjun and Shen, Li and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2410.07272},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Thanks

This repository is modified based on https://github.com/woodenchild95/FL-Simulator.


