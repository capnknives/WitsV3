# Ensure Docker Desktop is running before WitsV3 starts (sandbox_mode: docker).
# Safe to run repeatedly. Used by start_web_ui.bat and the watchdog.

param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [int]$WaitSeconds = 180
)

$ErrorActionPreference = "Stop"
$logFile = Join-Path $ProjectDir "logs\docker_companion.log"

function Write-Log($message) {
    $dir = Split-Path $logFile -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $message"
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Output $line
}

function Test-DockerDaemon {
    try {
        & docker info *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

if (Test-DockerDaemon) {
    Write-Log "Docker daemon already ready."
    exit 0
}

$desktop = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $desktop)) {
    Write-Log "ERROR - Docker Desktop not found at $desktop"
    exit 1
}

Write-Log "Starting Docker Desktop..."
Start-Process -FilePath $desktop -WindowStyle Hidden

$elapsed = 0
while ($elapsed -lt $WaitSeconds) {
    Start-Sleep -Seconds 3
    $elapsed += 3
    if (Test-DockerDaemon) {
        Write-Log "Docker daemon ready after ${elapsed}s."
        exit 0
    }
}

Write-Log "ERROR - Docker daemon not ready after ${WaitSeconds}s."
exit 1
