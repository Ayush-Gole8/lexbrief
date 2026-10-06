# Download raw LexBrief data (BUILD from Hugging Face, IN-Ext from Zenodo) and inspect it.
# Usage (from repo root):  .\scripts\download_data.ps1 [-Force]
# BUILD is gated: accept the terms at https://huggingface.co/datasets/opennyaiorg/InRhetoricalRoles
# and run `hf auth login` in the venv first.

param([switch]$Force)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Activate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "No .venv found. Run .\scripts\setup_env.ps1 first." -ForegroundColor Red
    exit 1
}
. $Activate

$dlArgs = @("download", "--config", "configs/data.yaml")
if ($Force) { $dlArgs += "--force" }

lexbrief @dlArgs
$dlExit = $LASTEXITCODE

lexbrief inspect-raw --config configs/data.yaml

if ($dlExit -ne 0) {
    Write-Host "Download incomplete; follow the manual instructions printed above." -ForegroundColor Yellow
    exit $dlExit
}
Write-Host "Raw data ready. Next: lexbrief prepare-data" -ForegroundColor Green
