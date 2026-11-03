
import sys
import flwr as fl
import numpy as np


CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def make_data(client_id):
    rng = np.random.default_rng(100 + client_id)

    # Different local data for each client
    X = rng.normal(
        loc=(client_id * 0.8, client_id * 0.4),
        scale=1.0,
        size=(500, 2),
    )

    # A shared underlying classification task
    y = (
        X[:, 0] + X[:, 1] > 0.8
    ).astype(np.float64)

    return X, y


class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.X, self.y = make_data(CLIENT_ID)
        self.w = np.zeros(2, dtype=np.float64)
        self.b = np.zeros(1, dtype=np.float64)

    def get_parameters(self, config):
        return [self.w.copy(), self.b.copy()]

    def fit(self, parameters, config):
        self.w, self.b = [p.copy() for p in parameters]

        # Train locally without sending raw data
        for _ in range(5):
            logits = self.X @ self.w + self.b[0]
            probs = 1.0 / (1.0 + np.exp(-np.clip(
                logits, -30, 30
            )))

            error = probs - self.y
            self.w -= 0.1 * (self.X.T @ error) / len(self.y)
            self.b -= 0.1 * np.array([error.mean()])

        print(f"Client {CLIENT_ID}: local training complete")
        return self.get_parameters({}), len(self.y), {}

    def evaluate(self, parameters, config):
        self.w, self.b = [p.copy() for p in parameters]

        logits = self.X @ self.w + self.b[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(
            logits, -30, 30
        )))

        loss = -np.mean(
            self.y * np.log(probs + 1e-9)
            + (1 - self.y) * np.log(1 - probs + 1e-9)
        )

        accuracy = np.mean((probs >= 0.5) == self.y)

        return float(loss), len(self.y), {
            "accuracy": float(accuracy)
        }


if __name__ == "__main__":
    client = FLClient()
    print(f"Connecting client {CLIENT_ID} to FL server...")
    fl.client.start_numpy_client(
        server_address="192.168.56.10:8080",
        client=client,
    )
