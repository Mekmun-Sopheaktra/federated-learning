import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


DATA_DIR = "/opt/fl-data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def load_train_data(client_id, batch_size=64):

    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    labels = dataset.targets

    if client_id == 1:
        selected_labels = [0, 1, 2, 3, 4]

    elif client_id == 2:
        selected_labels = [5, 6, 7, 8, 9]

    else:
        raise ValueError("Client ID must be 1 or 2")

    indices = torch.where(
        torch.isin(
            labels,
            torch.tensor(selected_labels)
        )
    )[0].tolist()

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return loader


def load_test_data(batch_size=256):

    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
