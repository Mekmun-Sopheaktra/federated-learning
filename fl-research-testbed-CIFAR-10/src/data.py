
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


DATA_DIR = "/opt/fl-data"

DATA_MODE = os.getenv(
    "FL_DATA_MODE", "non_iid"
).lower()


# CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)


# Training augmentation
train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)
])


# Testing: no augmentation
test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean, std)
])


def load_train_data(client_id, batch_size=64):

    if client_id not in [1, 2]:
        raise ValueError("Client ID must be 1 or 2")

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    labels = torch.tensor(dataset.targets)

    if DATA_MODE == "iid":

        generator = torch.Generator().manual_seed(42)

        shuffled_indices = torch.randperm(
            len(dataset),
            generator=generator
        ).tolist()

        midpoint = len(shuffled_indices) // 2

        partitions = [
            shuffled_indices[:midpoint],
            shuffled_indices[midpoint:]
        ]

        indices = partitions[client_id - 1]

    elif DATA_MODE == "non_iid":

        if client_id == 1:
            selected_labels = [0, 1, 2, 3, 4]

        else:
            selected_labels = [5, 6, 7, 8, 9]

        indices = torch.where(
            torch.isin(
                labels,
                torch.tensor(selected_labels)
            )
        )[0].tolist()

    else:
        raise ValueError(
            "FL_DATA_MODE must be 'iid' or 'non_iid'"
        )

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print(
        f"Client {client_id} | "
        f"Mode: {DATA_MODE} | "
        f"Samples: {len(subset)}"
    )

    return loader


def load_test_data(batch_size=256):

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
