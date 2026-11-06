
import os
import csv
import time

import flwr as fl
import torch
import torch.nn as nn

from model import CNN, set_parameters
from data import load_test_data


# ==========================================
# 1. Configuration
# ==========================================

DEVICE = torch.device("cpu")

NUM_ROUNDS = 10
NUM_CLIENTS = 2

FL_DATA_MODE = os.environ.get(
    "FL_DATA_MODE", "iid"
).strip().lower()

SUPPORTED_DATA_MODES = ["iid", "non_iid"]

if FL_DATA_MODE not in SUPPORTED_DATA_MODES:
    raise ValueError(
        f"Invalid FL_DATA_MODE: {FL_DATA_MODE}. "
        f"Supported modes: {SUPPORTED_DATA_MODES}"
    )

SERVER_ADDRESS = "0.0.0.0:8080"

RESULTS_DIR = "/vagrant/results"

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    f"results_{FL_DATA_MODE}.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. Initialize Dataset and Global Model
# ==========================================

print("\n========================================")
print("       FEDERATED LEARNING SERVER")
print("========================================")

print(f"Data Mode: {FL_DATA_MODE.upper()}")
print(f"Communication Rounds: {NUM_ROUNDS}")
print(f"Number of Clients: {NUM_CLIENTS}")
print(f"Device: {DEVICE}")
print(f"Results File: {RESULT_FILE}")

print("========================================\n")

test_loader = load_test_data()

model = CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()


# ==========================================
# 3. Experiment Metrics
# ==========================================

pending_metrics = {
    "training_seconds": 0.0,
    "aggregation_seconds": 0.0
}

CSV_FIELDS = [
    "round",
    "global_loss",
    "global_accuracy",
    "training_seconds",
    "aggregation_seconds",
    "model_size_bytes",
    "estimated_communication_bytes"
]


# Initialize a clean CSV for this experiment.
# Prevents old results from being mixed with new runs.

with open(
        RESULT_FILE,
        "w",
        newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=CSV_FIELDS
    )

    writer.writeheader()


# ==========================================
# 4. Global Model Evaluation
# ==========================================

def evaluate_global_model(
        server_round,
        parameters,
        config
):

    # Load aggregated model parameters
    set_parameters(model, parameters)

    model.eval()

    loss_total = 0.0
    correct = 0
    total = 0

    evaluation_start = time.perf_counter()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss_total += (
                    loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                    predictions == labels
            ).sum().item()

            total += labels.size(0)

    evaluation_seconds = (
            time.perf_counter() - evaluation_start
    )

    if total == 0:
        raise RuntimeError(
            "Global test dataset is empty."
        )

    average_loss = loss_total / total

    accuracy = correct / total


    # ======================================
    # Communication Cost Estimation
    # ======================================

    # Flower evaluate_fn receives a list of NumPy arrays.
    # Calculate the total serialized parameter size.

    model_bytes = sum(
        tensor.nbytes
        for tensor in parameters
    )

    # Two clients:
    # Server -> Client 1
    # Server -> Client 2
    # Client 1 -> Server
    # Client 2 -> Server

    communication_bytes = (
            model_bytes * NUM_CLIENTS * 2
    )


    # ======================================
    # Save Experiment Results
    # ======================================

    # Round 0 is the initial model evaluation.
    # It is not a completed federated communication round.
    # Therefore, only save rounds 1 through NUM_ROUNDS.

    if server_round > 0:

        with open(
                RESULT_FILE,
                "a",
                newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS
            )

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

                "estimated_communication_bytes": (
                    communication_bytes
                )
            })


    # ======================================
    # Display Evaluation Results
    # ======================================

    print("\n========== GLOBAL EVALUATION ==========")

    print(f"Data Mode: {FL_DATA_MODE.upper()}")

    print(f"Round: {server_round}/{NUM_ROUNDS}")

    print(f"Global Loss: {average_loss:.4f}")

    print(
        f"Global Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Training Time: "
        f"{pending_metrics['training_seconds']:.2f} seconds"
    )

    print(
        f"Aggregation Time: "
        f"{pending_metrics['aggregation_seconds']:.4f} seconds"
    )

    print(
        f"Evaluation Time: "
        f"{evaluation_seconds:.2f} seconds"
    )

    print(
        f"Model Size: "
        f"{model_bytes / 1024:.2f} KB"
    )

    print(
        f"Estimated Communication: "
        f"{communication_bytes / 1024:.2f} KB"
    )

    print("=======================================\n")

    return average_loss, {
        "accuracy": accuracy
    }


# ==========================================
# 5. Aggregate Client Training Metrics
# ==========================================

def aggregate_fit_metrics(metrics):

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    weighted_train_loss = sum(
        num_examples * values["train_loss"]
        for num_examples, values in metrics
    ) / total_examples

    weighted_train_seconds = sum(
        num_examples * values["train_seconds"]
        for num_examples, values in metrics
    ) / total_examples

    return {

        "train_loss": weighted_train_loss,

        "train_seconds": weighted_train_seconds

    }


# ==========================================
# 6. Custom FedAvg Strategy
# ==========================================

class ResearchFedAvg(fl.server.strategy.FedAvg):

    def aggregate_fit(
            self,
            server_round,
            results,
            failures
    ):

        start_time = time.perf_counter()

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )

        aggregation_time = (
                time.perf_counter() - start_time
        )

        pending_metrics[
            "aggregation_seconds"
        ] = aggregation_time

        if aggregated_metrics:

            pending_metrics[
                "training_seconds"
            ] = aggregated_metrics.get(
                "train_seconds", 0.0
            )

        else:

            pending_metrics[
                "training_seconds"
            ] = 0.0

        return (
            aggregated_parameters,
            aggregated_metrics
        )


# ==========================================
# 7. Main Server
# ==========================================

if __name__ == "__main__":

    print("Starting Federated Learning Server...")

    strategy = ResearchFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        evaluate_fn=evaluate_global_model,

        fit_metrics_aggregation_fn=aggregate_fit_metrics

    )

    fl.server.start_server(

        server_address=SERVER_ADDRESS,

        config=fl.server.ServerConfig(

            num_rounds=NUM_ROUNDS,

            round_timeout=600

        ),

        strategy=strategy

    )

    print("\nFederated Learning experiment completed.")

    print(f"Results saved to: {RESULT_FILE}")
