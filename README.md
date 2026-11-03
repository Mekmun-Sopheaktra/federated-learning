# Federated Learning (FL) Testbed

This project provides a small-scale **Federated Learning (FL) testbed** using Vagrant-based virtual machines to simulate a distributed learning environment. It consists of one global server and two client servers, allowing experimentation with local model training, model aggregation, and global model evaluation without sharing raw training data.

## 1. Federated Learning Vagrant Lab

Set up a Vagrant environment to simulate a federated learning architecture using multiple virtual machines.

**Technologies and libraries:**
- **Vagrant:** Creates and manages virtual machines.
- **Flower (FLwr):** Framework for implementing federated learning communication and coordination.
- **NumPy:** Supports numerical operations and data processing.
- **PyTorch:** Implements and trains the CNN model using MNIST.

### Infrastructure Architecture

The testbed consists of three virtual machines:

| Virtual Machine | Role | Responsibility |
|---|---|---|
| Global Server | FL Coordinator | Initializes the global model, coordinates training rounds, and aggregates client updates. |
| Client 1 | FL Client | Trains the local model using a restricted subset of MNIST classes. |
| Client 2 | FL Client | Trains the local model using a different subset of MNIST classes. |

## 2. Federated Learning Research Testbed

The research testbed is designed to investigate federated learning under **non-IID (non-independent and identically distributed) data conditions**.

Each client receives a different subset of the MNIST dataset, simulating a distributed environment where participating clients have different local data distributions.

### Experimental Configuration

- **Global Server:** Coordinates federated learning rounds and aggregates local model parameters using Federated Averaging (FedAvg).
- **Client 1:** Trains a local CNN model using a restricted subset of MNIST classes.
- **Client 2:** Trains a local CNN model using a different subset of MNIST classes.
- **Evaluation:** The global server evaluates the aggregated global model using the complete MNIST test dataset containing 10,000 samples.

### System Architecture

The following table summarizes the hardware virtualization environment, software frameworks, dataset, and experimental configurations used in this federated learning testbed.

| Component | Technology / Configuration |
|---|---|
| Host OS | Windows 10/11 |
| Virtualization | Oracle VirtualBox |
| VM Management | Vagrant |
| Guest OS | Ubuntu 22.04 LTS |
| FL Framework | Flower (FLwr) |
| Deep Learning Framework | PyTorch |
| Numerical Computing | NumPy |
| Dataset | MNIST |
| Model Architecture | Convolutional Neural Network (CNN) |
| Aggregation Algorithm | Federated Averaging (FedAvg) |
| Data Partitioning | Non-IID Label Distribution |
| Number of Virtual Machines | 3 (1 Global Server + 2 Clients) |
| Local Training Epochs | 2 |
| Communication Rounds | 10 |
| Evaluation Dataset | MNIST Test Set (10,000 samples) |
| Evaluation Metrics | Accuracy, Loss, Training Time, Communication Overhead (Bytes) |

## 3. Expected Experiment Workflow

The experiment follows an iterative federated learning process consisting of 10 communication rounds.

1. **Global Model Initialization:** The server initializes a global CNN model with randomly initialized parameters.
2. **Model Distribution:** The server sends the current global model parameters to both clients.
3. **Local Training:** Each client trains its local CNN model independently for two epochs using its own MNIST training subset.
4. **Model Updates:** Both clients return their updated model parameters and local training metrics to the server.
5. **Global Aggregation:** The server applies Federated Averaging (FedAvg) to aggregate the received client parameters and generate an updated global model.
6. **Global Evaluation:** The server evaluates the updated global model against the common MNIST test dataset (10,000 samples).
7. **Next Communication Round:** The updated global model is distributed to both clients, and the process repeats.

### Experiment Parameters

| Parameter | Configuration |
|---|---|
| Number of global servers | 1 |
| Number of clients | 2 |
| Dataset | MNIST |
| Model architecture | Convolutional Neural Network (CNN) |
| Local training epochs | 2 |
| Aggregation algorithm | FedAvg |
| Communication rounds | 10 |
| Global test samples | 10,000 |
| Data distribution | Non-IID |

## 4. Research Objectives

The primary objectives of this testbed are to:

- Understand the practical implementation of federated learning in a distributed virtual machine environment.
- Investigate the impact of non-IID data distributions on federated model performance.
- Implement and evaluate Federated Averaging (FedAvg).
- Monitor global model accuracy across communication rounds.
- Analyze the relationship between local training, model aggregation, and global convergence.

## 5. Expected Results

The experiment is expected to produce the following results:

- Global model accuracy and loss across 10 communication rounds.
- Local training loss and accuracy for each client.
- Comparison of client-specific data distributions.
- Evaluation of global model convergence under non-IID data conditions.

The results will provide a baseline for future experiments involving additional clients, alternative aggregation algorithms, and more challenging data distributions.