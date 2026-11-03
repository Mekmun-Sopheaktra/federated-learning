
import flwr as fl
import numpy as np


def get_initial_parameters():
    # Two features and one bias term
    return [
        np.zeros((2,), dtype=np.float64),
        np.zeros((1,), dtype=np.float64),
    ]


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_initial_parameters()
        ),
    )

    print("Starting Federated Learning server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
