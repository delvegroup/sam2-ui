Write-Host "=== Setting up SAM2 UI ===" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Yellow
} catch {
    Write-Host "Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    Exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Detect CUDA version
Write-Host "Detecting CUDA version..." -ForegroundColor Yellow
$cuda_version = $null

# Try nvcc first (most reliable)
try {
    $nvcc_output = nvcc --version 2>$null | Select-String "release (\d+)\.(\d+)"
    if ($nvcc_output -match "release (\d+)\.(\d+)") {
        $cuda_major = [int]$matches[1]
        $cuda_minor = [int]$matches[2]
        Write-Host "Found CUDA $cuda_major.$cuda_minor" -ForegroundColor Green
        
        # Map to PyTorch CUDA versions
        if ($cuda_major -eq 11) {
            $cuda_version = "cu118"
        } elseif ($cuda_major -eq 12) {
            # PyTorch cu121 works for CUDA 12.1-12.8
            $cuda_version = "cu121"
        }
    }
} catch {
    # Silently continue
}

# Fallback to nvidia-smi if nvcc failed
if (-not $cuda_version) {
    try {
        $nvidia_smi = nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "NVIDIA GPU detected via nvidia-smi, but nvcc not found" -ForegroundColor Yellow
            Write-Host "Assuming CUDA 12.x" -ForegroundColor Yellow
            $cuda_version = "cu121"
        }
    } catch {
        # No CUDA
    }
}

if (-not $cuda_version) {
    Write-Host "ERROR: CUDA not detected. SAM2 requires CUDA 11.8 or newer." -ForegroundColor Red
    Write-Host "Please install CUDA from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    Exit 1
}

Write-Host "Installing PyTorch with $cuda_version support..." -ForegroundColor Yellow
pip install torch torchvision --index-url https://download.pytorch.org/whl/$cuda_version

# Install other requirements
Write-Host "Installing requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# Clone and install SAM2
Write-Host "Installing SAM2 from GitHub..." -ForegroundColor Yellow
if (Test-Path "sam2_temp") {
    Remove-Item -Recurse -Force sam2_temp
}
git clone https://github.com/facebookresearch/sam2.git sam2_temp
Push-Location sam2_temp
pip install .
Pop-Location
Remove-Item -Recurse -Force sam2_temp

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next, run the UI: python .\run_ui.py"
Write-Host ""