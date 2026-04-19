# Activate the project venv in the current PowerShell session.
# Run from project root: . .\bin\activate.ps1
$projectRoot = Split-Path -Parent $PSScriptRoot
$activateScript = Join-Path $projectRoot 'venv\Scripts\Activate.ps1'
if (-not (Test-Path $activateScript)) {
    Write-Error "Venv not found. Run bin\init.bat first."
    exit 1
}
. $activateScript
