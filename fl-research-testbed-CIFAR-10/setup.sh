#!/bin/bash

set -e

echo "Preparing Federated Learning environment..."

# Install only missing system dependencies
export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get install -y \
    python3-venv \
    python3-pip

# Create virtual environment if it doesn't exist
if [ ! -d /opt/fl-venv ]; then
    python3 -m venv /opt/fl-venv
fi

# Install dependencies
/opt/fl-venv/bin/pip install --upgrade pip

/opt/fl-venv/bin/pip install \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

/opt/fl-venv/bin/pip install \
    -r /vagrant/requirements.txt

mkdir -p /opt/fl-data

echo "Federated Learning setup completed!"
