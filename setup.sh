#!/bin/bash

echo "=== Setting up SAM2 UI ==="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Detect CUDA version
echo "Detecting CUDA version..."
cuda_version=""

# Try nvcc first (most reliable)
if command -v nvcc &> /dev/null; then
    cuda_version_full=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    cuda_major=$(echo $cuda_version_full | cut -d'.' -f1)
    cuda_minor=$(echo $cuda_version_full | cut -d'.' -f2)
    
    echo "Found CUDA $cuda_major.$cuda_minor"
    
    if [ "$cuda_major" == "11" ]; then
        cuda_version="cu118"
    elif [ "$cuda_major" == "12" ]; then
        # PyTorch cu121 works for CUDA 12.1-12.8
        cuda_version="cu121"
    fi
fi

# Fallback to nvidia-smi if nvcc failed
if [ -z "$cuda_version" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected via nvidia-smi, but nvcc not found"
        echo "Assuming CUDA 12.x"
        cuda_version="cu121"
    fi
fi

if [ -z "$cuda_version" ]; then
    echo "ERROR: CUDA not detected. SAM2 requires CUDA 11.8 or newer."
    echo "Please install CUDA from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "Installing PyTorch with $cuda_version support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/$cuda_version

# Install other requirements
echo "Installing requirements.txt..."
pip install -r requirements.txt

# Clone and install SAM2
echo "Installing SAM2 from GitHub..."
if [ -d "sam2_temp" ]; then
    rm -rf sam2_temp
fi
git clone https://github.com/facebookresearch/sam2.git sam2_temp
pushd sam2_temp > /dev/null
pip install .
popd > /dev/null
rm -rf sam2_temp

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the UI:"
echo "   python run_ui.py"
echo ""