import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import load, save
from torch.utils.data import DataLoader
from nar_data import nar_Custom_Dataset
from torch.utils.data.dataset import random_split


def load_mnist(batch_size: int=64, root: str='./data/'):
    """
    Load MNIST data
    """
    t = transforms.Compose([
                       transforms.ToTensor(),
                       ]
                       )
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=t)
    test_data  = datasets.MNIST(root=root, train=False, download=True, transform=t)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, test_dataloader


def load_cifar(batch_size: int=64, root: str="./data/"):
    """
    Load CIFAR-10 data
    """
    transform_train = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
       
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    train_data, valid_data = random_split(train, [45000, 5000])
    test_data  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


def load_nar(num_task=10000, batch_size: int=64):

    try:
        nar_dataset_train = load("./data/nar_train.pkl")
        nar_dataset_test = load("./data/nar_test.pkl")
        print(f"Found NAR dataset at './data/nar_train.pkl' and './data/nar_test.pkl' :) and Loaded :)")
    except:
        print("Couldn't find NAR dataset :(")
        nar_dataset_train= nar_Custom_Dataset(num_task, train=True)
        nar_dataset_test= nar_Custom_Dataset(num_task, train=False)
        save("./data/nar_train.pkl", nar_dataset_train)
        save("./data/nar_test.pkl", nar_dataset_test)

    train_dataloader = DataLoader(nar_dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(nar_dataset_test, batch_size=batch_size, drop_last=True, shuffle=True)
    return train_dataloader, test_dataloader
    