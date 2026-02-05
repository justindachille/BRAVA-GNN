#!/bin/bash

if ! command -v uv &> /dev/null; then
    pip install uv
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

source .venv/bin/activate

echo "Checking dependencies..."

uv pip install torch==2.2.2

# Install pre-built binary to prevent source compilation issues
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

uv pip install fire pytorch-lightning "torch-geometric>=1.6.3" pandas

if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt..."
    uv pip install -r requirements.txt
fi

if [ -d "abcde" ]; then
    echo "Installing ABCDE..."
    uv pip install -e abcde --no-deps
fi