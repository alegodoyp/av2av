#!/bin/bash

# Exit on error
set -e

echo "=== Setting up AV2AV Pipeline for Ubuntu ==="

echo "[1/5] Updating package lists..."
sudo apt-get update -y

echo "[2/5] Installing system dependencies..."
sudo apt-get install -y ffmpeg build-essential python3-dev python3-venv git

echo "[3/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created in ./venv"
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch for CUDA (adjust version if needed for specific hardware)
# Using standard PyTorch command for Linux with CUDA 11.8 (or choose newest based on fairseq support)
# We assume fairseq supports standard recent torches or defaults.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[5/5] Installing fairseq from local folder..."
if [ -d "fairseq" ]; then
    pip install -e ./fairseq
else
    echo "WARNING: fairseq folder not found in current directory. Please clone it or ensure it's present."
    echo "If you need to clone it: git clone https://github.com/facebookresearch/fairseq"
fi

echo "=== Setup Complete! ==="
echo "To activate the environment, run:"
echo "source venv/bin/activate"
