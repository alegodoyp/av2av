#!/bin/bash

# Exit on error
set -e

echo "=== Setting up AV2AV Pipeline for Ubuntu (Conda version) ==="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.anaconda.com/free/miniconda/"
    exit 1
fi

ENV_NAME="av2av_env"

echo "[1/4] Creating Conda environment '$ENV_NAME' with Python, FFmpeg, and Git..."
# We use Python 3.10 (av2av recommends Python >=3.7,<3.11)
conda create -y -n $ENV_NAME -c conda-forge python=3.10 "ffmpeg<5" git

echo "[2/4] Activating environment..."
# Properly initialize conda configuration for the script block
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch for CUDA 11.8 (Adjust if your Linux box has a different CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[4/4] Installing fairseq from local folder..."
if [ -d "fairseq" ]; then
    pip install -e ./fairseq
else
    echo "WARNING: fairseq folder not found in current directory. Please clone it or ensure it's present."
    echo "git clone https://github.com/facebookresearch/fairseq"
fi

echo "=== Setup Complete! ==="
echo "To activate the environment in the future, run:"
echo "conda activate $ENV_NAME"
