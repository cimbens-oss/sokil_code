#!/bin/bash
# ============================================================
# FAST-LIVO2 Python Pipeline — Setup Script
# ============================================================
# Creates a virtual environment and installs all dependencies.
#
# Usage:
#   bash setup.sh
#
# After setup, run the pipeline with:
#   python run.py                     (interactive)
#   python run.py Bags/scan.bag       (direct)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  FAST-LIVO2 Python Pipeline — Setup"
echo "============================================================"
echo ""

# Create venv if not already in one
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating existing virtual environment (.venv/)..."
    else
        echo "Creating virtual environment (.venv/)..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    echo "  Python: $(which python3)"
    echo "  Version: $(python3 --version)"
    echo ""
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."
echo ""

# Verify imports
echo "Verifying imports..."
python3 -c "
import numpy, cv2, yaml, rosbags
print(f'  numpy:    {numpy.__version__}')
print(f'  opencv:   {cv2.__version__}')
print(f'  pyyaml:   {yaml.__version__}')
print(f'  rosbags:  {rosbags.__version__}')
print('  All dependencies OK.')
"
echo ""

echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To run the pipeline:"
echo "    source .venv/bin/activate"
echo "    python run.py"
echo ""
echo "  Or directly:"
echo "    .venv/bin/python run.py Bags/your_scan.bag"
echo "============================================================"
