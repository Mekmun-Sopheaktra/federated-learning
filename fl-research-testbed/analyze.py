
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Configuration
# ==========================================

RESULT_FILE = "results/federated_results.csv"
OUTPUT_DIR = "results/analysis"

TARGET_ACCURACY = 0.90

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Load experiment data
# ==========================================

def load_results():

    if not os.path.exists(RESULT_FILE):
        print(f"Error: Result file not found: {RESULT_FILE}")
        sys.exit(1)

    df = pd.read_csv(RESULT_FILE)

    if df.empty:
        print("Error: No experiment results found.")
        sys.exit(1)

    required_columns = [
        "round",
        "global_loss",
        "global_accuracy",
        "training_seconds",
        "aggregation_seconds",
        "model_size_bytes",
        "estimated_communication_bytes"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        print(f"Missing columns: {missing}")
        sys.exit(1)

    df = df.sort_values("round").reset_index(drop=True)

    print("\nFederated Learning Results Loaded")
    print("---------------------------------")
    print(f"Total rounds: {len(df)}")
    print(f"First round: {df['round'].min()}")
    print(f"Last round: {df['round'].max()}")

    return df


# ==========================================
# Plot helper
# ==========================================

def save_plot(filename):

    path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


# ==========================================
# 1. Global Accuracy
# ==========================================

def plot_accuracy(df):

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        df["global_accuracy"] * 100,
        marker="o",
        linewidth=2,
        label="Global Accuracy"
    )

    plt.axhline(
        TARGET_ACCURACY * 100,
        linestyle="--",
        label="Target Accuracy (90%)"
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Global Accuracy")

    plt.grid(True)
    plt.legend()

    save_plot("accuracy.png")


# ==========================================
# 2. Global Loss
# ==========================================

def plot_loss(df):

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        df["global_loss"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Federated Learning Global Loss")

    plt.grid(True)

    save_plot("loss.png")


# ==========================================
# 3. Training Time
# ==========================================

def plot_training_time(df):

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        df["training_seconds"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Training Time (seconds)")
    plt.title("Client Training Time per Round")

    plt.grid(True)

    save_plot("training_time.png")


# ==========================================
# 4. Aggregation Time
# ==========================================

def plot_aggregation_time(df):

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        df["aggregation_seconds"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Aggregation Time (seconds)")
    plt.title("Server Aggregation Time")

    plt.grid(True)

    save_plot("aggregation_time.png")


# ==========================================
# 5. Communication Cost
# ==========================================

def plot_communication_cost(df):

    communication_mb = (
        df["estimated_communication_bytes"] / (1024 ** 2)
    )

    cumulative_mb = communication_mb.cumsum()

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        cumulative_mb,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Cumulative Communication (MB)")
    plt.title("Federated Learning Communication Overhead")

    plt.grid(True)

    save_plot("communication_cost.png")


# ==========================================
# 6. Model Size
# ==========================================

def plot_model_size(df):

    model_kb = df["model_size_bytes"] / 1024

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["round"],
        model_kb,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Model Size (KB)")
    plt.title("Global Model Communication Size")

    plt.grid(True)

    save_plot("model_size.png")


# ==========================================
# 7. Calculate Research Metrics
# ==========================================

def calculate_metrics(df):

    best_accuracy = df["global_accuracy"].max()

    best_round = int(
        df.loc[
            df["global_accuracy"].idxmax(),
            "round"
        ]
    )

    final_accuracy = df.iloc[-1]["global_accuracy"]

    final_loss = df.iloc[-1]["global_loss"]

    avg_training = df["training_seconds"].mean()

    avg_aggregation = df["aggregation_seconds"].mean()

    total_communication = (
        df["estimated_communication_bytes"].sum()
    )

    convergence = df[
        df["global_accuracy"] >= TARGET_ACCURACY
    ]

    convergence_round = (
        int(convergence.iloc[0]["round"])
        if not convergence.empty
        else None
    )

    total_training_time = df["training_seconds"].sum()

    total_aggregation_time = df["aggregation_seconds"].sum()

    metrics = {
        "Total Communication Rounds": len(df),
        "Final Global Accuracy": f"{final_accuracy * 100:.2f}%",
        "Best Global Accuracy": f"{best_accuracy * 100:.2f}%",
        "Best Accuracy Round": best_round,
        "Final Global Loss": f"{final_loss:.4f}",
        "Average Training Time": f"{avg_training:.2f} seconds",
        "Average Aggregation Time": f"{avg_aggregation:.4f} seconds",
        "Total Estimated Communication": f"{total_communication / (1024 ** 2):.4f} MB",
        "Total Recorded Training Time": f"{total_training_time:.2f} seconds",
        "Total Aggregation Time": f"{total_aggregation_time:.4f} seconds",
        "90% Accuracy Convergence Round": (
            convergence_round
            if convergence_round is not None
            else "Not reached"
        )
    }

    return metrics


# ==========================================
# 8. Export Summary
# ==========================================

def export_summary(metrics):

    path = os.path.join(OUTPUT_DIR, "summary.txt")

    with open(path, "w", encoding="utf-8") as file:

        file.write("FEDERATED LEARNING RESEARCH REPORT\n")
        file.write("=" * 45 + "\n\n")

        for key, value in metrics.items():

            file.write(f"{key}: {value}\n")

    print(f"Saved: {path}")


# ==========================================
# Main
# ==========================================

def main():

    df = load_results()

    print("\nGenerating research graphs...")

    plot_accuracy(df)
    plot_loss(df)
    plot_training_time(df)
    plot_aggregation_time(df)
    plot_communication_cost(df)
    plot_model_size(df)

    metrics = calculate_metrics(df)

    export_summary(metrics)

    print("\nRESEARCH PERFORMANCE SUMMARY")
    print("=" * 45)

    for key, value in metrics.items():

        print(f"{key}: {value}")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
