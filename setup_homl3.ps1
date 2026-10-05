# Hands-On Machine Learning 3rd Edition - Setup Script
# This script will set up the conda environment for the HOML3 project

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Hands-On ML 3rd Edition Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
$condaPath = $null
$possiblePaths = @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\AppData\Local\miniconda3\Scripts\conda.exe"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $condaPath = $path
        break
    }
}

if (-not $condaPath) {
    Write-Host "Conda not found! Please install Miniconda first:" -ForegroundColor Red
    Write-Host ""
    Write-Host "1. Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    Write-Host "2. Run the installer (use default settings)" -ForegroundColor Yellow
    Write-Host "3. Re-run this script from Anaconda Prompt" -ForegroundColor Yellow
    Write-Host ""
    Start-Process "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

Write-Host "Found conda at: $condaPath" -ForegroundColor Green
Write-Host ""

# Initialize conda for PowerShell if not already done
Write-Host "Step 1: Initializing conda for PowerShell..." -ForegroundColor Yellow
& $condaPath init powershell

# Create the homl3 environment
Write-Host ""
Write-Host "Step 2: Creating 'homl3' conda environment..." -ForegroundColor Yellow
Write-Host "This may take 10-20 minutes. Please be patient!" -ForegroundColor Cyan
& $condaPath env create -f environment.yml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment may already exist. Updating instead..." -ForegroundColor Yellow
    & $condaPath env update -f environment.yml
}

# Activate and register kernel
Write-Host ""
Write-Host "Step 3: Registering Jupyter kernel..." -ForegroundColor Yellow
& $condaPath run -n homl3 python -m ipykernel install --user --name=python3

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start learning, run these commands:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  conda activate homl3" -ForegroundColor White
Write-Host "  jupyter notebook" -ForegroundColor White
Write-Host ""
Write-Host "Or use JupyterLab (modern interface):" -ForegroundColor Cyan
Write-Host ""
Write-Host "  conda activate homl3" -ForegroundColor White
Write-Host "  jupyter lab" -ForegroundColor White
Write-Host ""
