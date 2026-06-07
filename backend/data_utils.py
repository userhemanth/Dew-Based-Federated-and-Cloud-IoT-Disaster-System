# utils/data_utils.py
"""
Data utilities for the Dew-Based Federated Learning disaster system.
Loads real disaster image data (ImageFolder format) instead of MNIST.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
import random

IMG_SIZE   = 224
BATCH_SIZE = 32

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_data(data_dir="data"):
    """
    Load the full disaster image dataset.
    Returns trainloader, testloader, and class names.
    Splits 80% train / 20% test.
    """
    train_dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)
    eval_dataset  = datasets.ImageFolder(root=data_dir, transform=EVAL_TRANSFORM)
    class_names   = train_dataset.classes

    n_total = len(train_dataset)
    n_train = int(0.8 * n_total)
    n_test  = n_total - n_train

    train_indices = list(range(n_train))
    test_indices  = list(range(n_train, n_total))

    trainset = torch.utils.data.Subset(train_dataset, train_indices)
    testset  = torch.utils.data.Subset(eval_dataset,  test_indices)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    testloader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[DataUtils] Loaded {len(class_names)} classes: {class_names}")
    print(f"[DataUtils] Train: {len(trainset)} | Test: {len(testset)}")
    return trainloader, testloader, class_names


def get_stratified_client_data(data_dir, client_id, num_clients=3):
    """
    Returns stratified (train, test) DataLoaders for a specific client.
    Every client sees all classes (IID-balanced split).
    """
    train_dataset = datasets.ImageFolder(root=data_dir, transform=TRAIN_TRANSFORM)
    eval_dataset  = datasets.ImageFolder(root=data_dir, transform=EVAL_TRANSFORM)
    class_names   = train_dataset.classes

    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset.samples):
        class_to_indices[label].append(idx)

    client_indices = []
    for indices in class_to_indices.values():
        shuffled = indices.copy()
        random.shuffle(shuffled)
        client_indices += shuffled[(client_id - 1)::num_clients]

    random.shuffle(client_indices)
    n_train = int(0.8 * len(client_indices))
    train_idx = client_indices[:n_train]
    test_idx  = client_indices[n_train:]

    trainset    = torch.utils.data.Subset(train_dataset, train_idx)
    testset     = torch.utils.data.Subset(eval_dataset,  test_idx)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    testloader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return trainloader, testloader, class_names
