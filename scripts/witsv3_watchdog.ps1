# WitsV3 watchdog — checks whether the web UI is listening on its configured
# port, and relaunches it via start_web_ui.bat if not. Safe to run repeatedly:
# run_web.py already detects an existing instance on the port and exits
# immediately instead of double-launching.
#
# Intended to run on a schedule (see docs/watchdog-and-wake-timer.md for the
# Register-ScheduledTask commands that wire this up):
#   - every N minutes, to relaunch after a crash
#   - once a few minutes before the daily self-repair cron (self_repair.daily_schedule_cron
#     in config.yaml, default 3am), combined with a wake-enabled task so a
#     sleeping PC wakes up first

param(
    [int]$Port = 8000,
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$logFile = Join-Path $ProjectDir "logs\watchdog.log"

function Write-Log($message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $message"
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Output $line
}

$listening = $false
try {
    $listening = (Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet)
} catch {
    $listening = $false
}

if ($listening) {
    Write-Log "OK - port $Port already has a listener, nothing to do."
    exit 0
}

Write-Log "Port $Port is not listening - relaunching WitsV3 web UI."
$batPath = Join-Path $ProjectDir "start_web_ui.bat"
if (-not (Test-Path $batPath)) {
    Write-Log "ERROR - $batPath not found; cannot relaunch."
    exit 1
}

Start-Process -FilePath $batPath -WorkingDirectory $ProjectDir -WindowStyle Minimized
Write-Log "Relaunch triggered via $batPath."
