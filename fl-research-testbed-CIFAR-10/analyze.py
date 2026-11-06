
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. Configuration
# ==========================================

FL_DATA_MODE = os.environ.get(
    "FL_DATA_MODE", "iid"
).strip().lower()

if FL_DATA_MODE not in ("iid", "non_iid"):
    raise ValueError(
        "FL_DATA_MODE must be either 'iid' or 'non_iid'."
    )

DATASET_NAME = os.environ.get(
    "FL_DATASET", "CIFAR-10"
)

TARGET_ACCURACY = 0.90

RESULTS_DIR = Path("results")

# Must match server.py filename.
RESULT_FILE = RESULTS_DIR / f"results_{FL_DATA_MODE}.csv"

OUTPUT_DIR = (
        RESULTS_DIR / "analysis" / FL_DATA_MODE
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = FL_DATA_MODE.upper().replace("_", "-")


# Required CSV columns
REQUIRED_COLUMNS = [
    "round",
    "global_loss",
    "global_accuracy",
    "training_seconds",
    "aggregation_seconds",
    "model_size_bytes",
    "estimated_communication_bytes"
]


# ==========================================
# 2. Load Experiment Results
# ==========================================

def load_results():

    print("\n" + "=" * 55)
    print("       FEDERATED LEARNING DATA ANALYSIS")
    print("=" * 55)

    print(f"Experiment Mode : {EXPERIMENT_NAME}")
    print(f"Dataset         : {DATASET_NAME}")
    print(f"Result File     : {RESULT_FILE}")

    if not RESULT_FILE.exists():

        print("\nERROR: Experiment result file not found.")
        print(f"Expected: {RESULT_FILE}")

        print("\nAvailable result files:")

        if RESULTS_DIR.exists():

            for file in RESULTS_DIR.glob("*.csv"):
                print(f"  - {file}")

        sys.exit(1)

    try:
        df = pd.read_csv(RESULT_FILE)

    except Exception as error:

        print(f"\nERROR: Cannot read CSV: {error}")
        sys.exit(1)

    if df.empty:

        print("\nERROR: Experiment result file is empty.")
        sys.exit(1)

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing required columns:")
        print(missing_columns)

        sys.exit(1)

    # Convert columns to numeric
    for column in REQUIRED_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Check invalid values
    if df[REQUIRED_COLUMNS].isnull().any().any():

        print("\nERROR: Invalid or missing numeric values.")

        print(
            df[
                df[REQUIRED_COLUMNS].isnull().any(axis=1)
            ]
        )

        sys.exit(1)

    # Sort by communication round
    df = df.sort_values(
        "round"
    ).reset_index(drop=True)

    # Check duplicate rounds
    if df["round"].duplicated().any():

        print("\nERROR: Duplicate communication rounds detected.")
        sys.exit(1)

    if (df["round"] < 1).any():

        print("\nERROR: Communication rounds must be positive.")
        sys.exit(1)

    # Accuracy should be represented between 0 and 1.
    if not df["global_accuracy"].between(0, 1).all():

        print(
            "\nERROR: global_accuracy must be between 0 and 1."
        )

        sys.exit(1)

    # Check non-negative metrics
    nonnegative_columns = [
        "global_loss",
        "training_seconds",
        "aggregation_seconds",
        "model_size_bytes",
        "estimated_communication_bytes"
    ]

    for column in nonnegative_columns:

        if (df[column] < 0).any():

            print(
                f"\nERROR: Negative values found in {column}."
            )

            sys.exit(1)

    print("\nExperiment Results Loaded Successfully")
    print("-" * 55)

    print(f"Total Rounds    : {len(df)}")
    print(f"First Round     : {int(df['round'].min())}")
    print(f"Last Round      : {int(df['round'].max())}")
    print(
        f"Final Accuracy  : "
        f"{df.iloc[-1]['global_accuracy'] * 100:.2f}%"
    )

    return df


# ==========================================
# 3. Plot Helper
# ==========================================

def save_plot(filename):

    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ==========================================
# 4. Global Accuracy
# ==========================================

def plot_accuracy(df):

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        df["global_accuracy"] * 100,
        marker="o",
        linewidth=2,
        markersize=6,
        label=f"{EXPERIMENT_NAME} Accuracy"
    )

    plt.axhline(
        TARGET_ACCURACY * 100,
        linestyle="--",
        linewidth=1.5,
        label=f"Target Accuracy ({TARGET_ACCURACY * 100:.0f}%)"
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Global Accuracy (%)")

    plt.title(
        f"Global Model Accuracy - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)
    plt.legend()

    save_plot("accuracy.png")


# ==========================================
# 5. Global Loss
# ==========================================

def plot_loss(df):

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        df["global_loss"],
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Communication Round")
    plt.ylabel("Cross-Entropy Loss")

    plt.title(
        f"Global Model Loss - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)

    save_plot("loss.png")


# ==========================================
# 6. Client Training Time
# ==========================================

def plot_training_time(df):

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        df["training_seconds"],
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Communication Round")

    plt.ylabel(
        "Sample-Weighted Client Training Time (seconds)"
    )

    plt.title(
        f"Client Training Time - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)

    save_plot("training_time.png")


# ==========================================
# 7. Server Aggregation Time
# ==========================================

def plot_aggregation_time(df):

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        df["aggregation_seconds"],
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Aggregation Time (seconds)")

    plt.title(
        f"Server Aggregation Time - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)

    save_plot("aggregation_time.png")


# ==========================================
# 8. Communication Cost
# ==========================================

def plot_communication_cost(df):

    communication_mb = (
            df["estimated_communication_bytes"]
            / (1024 ** 2)
    )

    cumulative_mb = communication_mb.cumsum()

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        cumulative_mb,
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Cumulative Communication (MB)")

    plt.title(
        f"Cumulative Communication Cost - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)

    save_plot("communication_cost.png")


# ==========================================
# 9. Global Model Size
# ==========================================

def plot_model_size(df):

    model_kb = (
            df["model_size_bytes"] / 1024
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["round"],
        model_kb,
        marker="o",
        linewidth=2,
        markersize=6
    )

    plt.xlabel("Communication Round")

    plt.ylabel("Model Size (KB)")

    plt.title(
        f"Global Model Size - {EXPERIMENT_NAME}"
    )

    plt.xticks(df["round"])

    plt.grid(True, alpha=0.3)

    save_plot("model_size.png")


# ==========================================
# 10. Calculate Research Metrics
# ==========================================

def calculate_metrics(df):

    # Accuracy
    best_accuracy = df["global_accuracy"].max()

    best_round = int(
        df.loc[
            df["global_accuracy"].idxmax(),
            "round"
        ]
    )

    final_accuracy = df.iloc[-1]["global_accuracy"]

    final_loss = df.iloc[-1]["global_loss"]

    # Training performance
    average_training = df[
        "training_seconds"
    ].mean()

    # Aggregation performance
    average_aggregation = df[
        "aggregation_seconds"
    ].mean()

    # Communication
    total_communication = df[
        "estimated_communication_bytes"
    ].sum()

    average_communication = df[
        "estimated_communication_bytes"
    ].mean()

    # Model size
    average_model_size = df[
        "model_size_bytes"
    ].mean()

    # Time to reach target accuracy
    convergence = df[
        df["global_accuracy"] >= TARGET_ACCURACY
        ]

    convergence_round = (
        int(convergence.iloc[0]["round"])
        if not convergence.empty
        else None
    )

    # Recorded training and aggregation times
    cumulative_recorded_training = df[
        "training_seconds"
    ].sum()

    cumulative_aggregation = df[
        "aggregation_seconds"
    ].sum()

    # Round-to-round accuracy improvement
    initial_accuracy = df.iloc[0]["global_accuracy"]

    accuracy_improvement = (
                                   final_accuracy - initial_accuracy
                           ) * 100

    metrics = {

        "Experiment Mode": EXPERIMENT_NAME,

        "Dataset": DATASET_NAME,

        "Total Communication Rounds": len(df),

        "Final Global Accuracy": (
            f"{final_accuracy * 100:.2f}%"
        ),

        "Best Global Accuracy": (
            f"{best_accuracy * 100:.2f}%"
        ),

        "Best Accuracy Round": best_round,

        "Final Global Loss": (
            f"{final_loss:.4f}"
        ),

        "Accuracy Improvement": (
            f"{accuracy_improvement:.2f} percentage points"
        ),

        "Average Client Training Time": (
            f"{average_training:.2f} seconds"
        ),

        "Average Server Aggregation Time": (
            f"{average_aggregation:.4f} seconds"
        ),

        "Cumulative Recorded Training Time": (
            f"{cumulative_recorded_training:.2f} seconds"
        ),

        "Cumulative Aggregation Time": (
            f"{cumulative_aggregation:.4f} seconds"
        ),

        "Average Model Size": (
            f"{average_model_size / 1024:.2f} KB"
        ),

        "Average Communication Per Round": (
            f"{average_communication / (1024 ** 2):.4f} MB"
        ),

        "Total Estimated Communication": (
            f"{total_communication / (1024 ** 2):.4f} MB"
        ),

        f"{TARGET_ACCURACY * 100:.0f}% Accuracy Round": (
            convergence_round
            if convergence_round is not None
            else "Not reached"
        )
    }

    return metrics


# ==========================================
# 11. Export Research Summary
# ==========================================

def export_summary(metrics):

    # Export TXT
    txt_path = OUTPUT_DIR / (
        f"{FL_DATA_MODE}_summary.txt"
    )

    with open(
            txt_path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(
            "FEDERATED LEARNING RESEARCH REPORT\n"
        )

        file.write("=" * 55 + "\n\n")

        for key, value in metrics.items():

            file.write(
                f"{key}: {value}\n"
            )

    print(f"Saved: {txt_path}")

    # Export CSV
    csv_path = OUTPUT_DIR / (
        f"{FL_DATA_MODE}_summary.csv"
    )

    pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    ).to_csv(
        csv_path,
        index=False
    )

    print(f"Saved: {csv_path}")


# ==========================================
# 12. Main
# ==========================================

def main():

    df = load_results()

    print("\nGenerating Research Graphs...")
    print("-" * 55)

    plot_accuracy(df)

    plot_loss(df)

    plot_training_time(df)

    plot_aggregation_time(df)

    plot_communication_cost(df)

    plot_model_size(df)

    # Calculate metrics
    metrics = calculate_metrics(df)

    # Export reports
    export_summary(metrics)

    # Display summary
    print("\n" + "=" * 55)
    print("       RESEARCH PERFORMANCE SUMMARY")
    print("=" * 55)

    for key, value in metrics.items():

        print(f"{key}: {value}")

    print("\n" + "=" * 55)
    print("Analysis completed successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":

    main()
