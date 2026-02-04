#!/bin/bash

# Ensure uv is available
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

# 1. Install PyTorch
uv pip install torch==2.2.2

# 2. Install PyG Binary Dependencies (CRITICAL FIX)
# We include torch-spline-conv here so it installs the pre-built binary.
# This prevents pip from trying to compile it from source (which caused your error).
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# 3. Install ABCDE Requirements
uv pip install fire pytorch-lightning "torch-geometric>=1.6.3" pandas

# 4. Install Project Requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt..."
    uv pip install -r requirements.txt
fi

# 5. Install ABCDE in Editable Mode
# Since torch-spline-conv is already installed above, this won't try to build it.
if [ -d "abcde" ]; then
    echo "Installing ABCDE..."
    uv pip install -e abcde --no-deps
fi