import sys
import time

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN, get_parameters, set_parameters
from data import load_train_data, load_test_data


DEVICE = torch.device("cpu")
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64


class FederatedClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id
        self.model = CNN().to(DEVICE)

        self.train_loader = load_train_data(
            client_id,
            BATCH_SIZE
        )

        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):

        return get_parameters(self.model)

    def fit(self, parameters, config):

        # Receive global model
        set_parameters(self.model, parameters)

        self.model.train()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        start_time = time.perf_counter()

        total_loss = 0.0

        for epoch in range(LOCAL_EPOCHS):

            epoch_loss = 0.0

            for images, labels in self.train_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss / len(self.train_loader)

        training_time = time.perf_counter() - start_time

        print(
            f"Client {self.client_id} | "
            f"Training completed | "
            f"Loss: {total_loss / LOCAL_EPOCHS:.4f} | "
            f"Time: {training_time:.2f}s"
        )

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "client_id": self.client_id,
                "train_loss": total_loss / LOCAL_EPOCHS,
                "train_seconds": training_time
            }
        )

    def evaluate(self, parameters, config):

        set_parameters(self.model, parameters)

        self.model.eval()

        test_loader = load_test_data()

        loss_total = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                loss_total += loss.item() * labels.size(0)

                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

        return (
            loss_total / total,
            total,
            {
                "accuracy": correct / total
            }
        )


if __name__ == "__main__":

    client_id = int(sys.argv[1])

    client = FederatedClient(client_id)

    print(
        f"Starting Client {client_id} "
        f"at 192.168.56.{10 + client_id}"
    )

    fl.client.start_numpy_client(
        server_address="192.168.56.10:8080",
        client=client
    )
