import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_parameters(model):

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(model, parameters):

    state_dict = model.state_dict()

    updated_state = {
        key: torch.tensor(value)
        for key, value in zip(state_dict.keys(), parameters)
    }

    model.load_state_dict(updated_state)
