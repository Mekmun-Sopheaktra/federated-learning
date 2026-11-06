
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

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

    updated_state = {}

    for key, value in zip(state_dict.keys(), parameters):

        updated_state[key] = torch.as_tensor(
            value,
            dtype=state_dict[key].dtype,
            device=state_dict[key].device
        )

    model.load_state_dict(updated_state)
