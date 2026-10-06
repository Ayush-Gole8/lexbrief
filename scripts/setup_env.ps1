# LexBrief environment setup (Windows / PowerShell).
# Usage (from repo root):  .\scripts\setup_env.ps1
# If blocked by execution policy:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    # Native tools (pip, spacy) write warnings to stderr; judge success by exit code only.
    $global:LASTEXITCODE = 0
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "SETUP FAILED at step: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Fix the error above and re-run .\scripts\setup_env.ps1" -ForegroundColor Red
        exit 1
    }
}

try {
    $Activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $Activate)) {
        Invoke-Step "Create .venv with Python 3.11" { py -3.11 -m venv .venv }
    }
    Write-Host "==> Activate .venv" -ForegroundColor Cyan
    . $Activate

    Invoke-Step "Check Python version" {
        python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)"
    }
    Invoke-Step "Upgrade pip" { python -m pip install --upgrade pip setuptools wheel }
    # torch >= 2.6 is required: transformers 5.x refuses to load .bin checkpoints (InLegalBERT)
    # on older torch (CVE-2025-32434). cu121 stops at 2.5.1, so use cu126 wheels; they run on
    # CUDA 12.x drivers via minor-version compatibility.
    Invoke-Step "Install torch >= 2.6 (CUDA 12.6 wheels)" {
        python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu126
    }
    Invoke-Step "Install requirements.txt" { python -m pip install -r requirements.txt }
    Invoke-Step "Install lexbrief (editable)" { python -m pip install -e . }
    Invoke-Step "Download spaCy en_core_web_sm" { python -m spacy download en_core_web_sm }
    Invoke-Step "Install pre-commit hooks" { pre-commit install }
    Invoke-Step "Verify CUDA torch" {
        python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print(torch.__version__, torch.cuda.get_device_name(0))"
    }
}
catch {
    Write-Host ""
    Write-Host "SETUP FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Setup complete. Next: lexbrief check-env" -ForegroundColor Green
