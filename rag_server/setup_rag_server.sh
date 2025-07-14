#!/bin/bash
set -e

# Optional: use Python 3.10+ if not default
PYTHON_BIN=python3
VENV_DIR=/opt/rag_server_venv
APP_DIR=/opt/rag_server
REQUIREMENTS=requirements.txt
CONSTRAINTS=constraints.txt

# Create virtual environment
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Copy application (adjust this step as needed)
sudo mkdir -p "$APP_DIR"
sudo cp -r ~/workrag-system-1/rag_server/* "$APP_DIR"

cd "$APP_DIR"

# Install latest available PyTorch for CUDA 12.1
pip install \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121



# Install app dependencies
pip install -r "$REQUIREMENTS"

# Optional check
python -c "import torch; print(torch.cuda.is_available())"

