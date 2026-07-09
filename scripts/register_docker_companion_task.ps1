# Register a logon task so Docker Desktop starts with Windows (companion to WitsV3).
# Run once as admin if desired; Wits also starts Docker on demand via ensure_docker_desktop.ps1.

$ErrorActionPreference = "Stop"
$script = Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\ensure_docker_desktop.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "WitsV3 Docker Companion" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force
Write-Host "Registered scheduled task: WitsV3 Docker Companion (At logon)"
