import os
import csv
import time

import flwr as fl
import torch
import torch.nn as nn

from model import CNN, get_parameters, set_parameters
from data import load_test_data


DEVICE = torch.device("cpu")
NUM_ROUNDS = 10

RESULTS_DIR = "/vagrant/results"
RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "federated_results.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

test_loader = load_test_data()

model = CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()

pending_metrics = {
    "training_seconds": 0.0,
    "aggregation_seconds": 0.0
}


def evaluate_global_model(server_round, parameters, config):

    set_parameters(model, parameters)

    model.eval()

    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = loss_total / total
    accuracy = correct / total

    # Approximate model communication volume:
    # Server -> 2 clients and 2 clients -> Server
    model_bytes = sum(
        parameter.nbytes
        for parameter in parameters
    )

    communication_bytes = model_bytes * 4

    # Append experiment results
    file_exists = os.path.exists(RESULT_FILE)

    with open(RESULT_FILE, "a", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "round",
                "global_loss",
                "global_accuracy",
                "training_seconds",
                "aggregation_seconds",
                "model_size_bytes",
                "estimated_communication_bytes"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "round": server_round,
            "global_loss": average_loss,
            "global_accuracy": accuracy,
            "training_seconds": pending_metrics[
                "training_seconds"
            ],
            "aggregation_seconds": pending_metrics[
                "aggregation_seconds"
            ],
            "model_size_bytes": model_bytes,
            "estimated_communication_bytes": communication_bytes
        })

    print("\n========== GLOBAL EVALUATION ==========")

    print(f"Round: {server_round}")
    print(f"Global Loss: {average_loss:.4f}")
    print(f"Global Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Estimated Communication: "
        f"{communication_bytes / 1024:.2f} KB"
    )

    print("=======================================\n")

    return average_loss, {
        "accuracy": accuracy
    }


def aggregate_fit_metrics(metrics):

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    return {
        "train_loss": sum(
            num_examples * values["train_loss"]
            for num_examples, values in metrics
        ) / total_examples,

        "train_seconds": sum(
            num_examples * values["train_seconds"]
            for num_examples, values in metrics
        ) / total_examples
    }


class ResearchFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        start_time = time.perf_counter()

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        aggregation_time = time.perf_counter() - start_time

        pending_metrics["aggregation_seconds"] = aggregation_time

        if aggregated_metrics:

            pending_metrics["training_seconds"] = (
                aggregated_metrics.get("train_seconds", 0.0)
            )

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":

    print("Starting Federated Learning Server...")

    strategy = ResearchFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=2,

        min_evaluate_clients=2,

        min_available_clients=2,

        evaluate_fn=evaluate_global_model,

        fit_metrics_aggregation_fn=aggregate_fit_metrics
    )

    fl.server.start_server(

        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS,
            round_timeout=600
        ),

        strategy=strategy
    )
