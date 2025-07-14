# Segment Anything 2 UI

This repository provides a simple Gradio UI for Meta's Segment Anything 2. On Windows, run `setup.ps1` in PowerShell. On Linux, run `setup.sh`.

The setup script will:
1. Create a venv
2. Upgrade pip
3. Install PyTorch for CUDA 12.4
4. Run `pip install -r requirements.txt`

After completing setup, run `run_ui.py` in Python.