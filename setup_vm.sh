#!/bin/bash
set -e

echo "=== Expedia Ranking VM Setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install deps
echo "Installing dependencies..."
uv sync

# Verify data exists
if [ ! -f "data/training_set_VU_DM.csv" ]; then
    echo ""
    echo "WARNING: Training data not found at data/training_set_VU_DM.csv"
    echo "Copy your data files:"
    echo "  scp data/training_set_VU_DM.csv <vm>:$(pwd)/data/"
    echo "  scp data/test_set_VU_DM.csv <vm>:$(pwd)/data/"
    echo ""
fi

# Create dirs
mkdir -p models/v4 submissions

# Quick smoke test
echo "Running smoke test..."
uv run python -c "
import lightgbm, pandas, numpy
print(f'LightGBM {lightgbm.__version__}')
print(f'pandas {pandas.__version__}')
print(f'numpy {numpy.__version__}')
from src.features import build_features, FORBIDDEN_FEATURES
print(f'Features module OK, {len(FORBIDDEN_FEATURES)} forbidden cols')
print('All good!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the pipeline:"
echo "  uv run python run_v4.py"
echo ""
echo "To run just Stage 1 sanity (fast):"
echo "  uv run python -c 'from run_v4 import stage1_sanity; stage1_sanity()'"
