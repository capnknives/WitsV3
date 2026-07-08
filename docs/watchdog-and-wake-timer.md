# Keeping WitsV3 running + waking the PC for the daily self-repair run

Two Windows Scheduled Tasks, both driving `scripts/witsv3_watchdog.ps1`
(checks whether port 8000 has a listener; if not, relaunches
`start_web_ui.bat` — safe to run repeatedly since `run_web.py` already exits
immediately if something's already listening).

## 1. Crash recovery — relaunch if the process died

Runs the watchdog every 10 minutes and once at logon, so a crashed
`run_web.py` process gets relaunched without you noticing.

```powershell
$action   = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Users\capta\source\repos\capnknives\WitsV3-claude\scripts\witsv3_watchdog.ps1"'
$trigger1 = New-ScheduledTaskTrigger -AtLogOn
$trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBattery -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "WitsV3 Watchdog" -Action $action -Trigger $trigger1,$trigger2 -Settings $settings -RunLevel Highest
```

## 2. Wake the PC before the 3am self-repair run

`self_repair.daily_schedule_cron` in `config.yaml` defaults to `0 3 * * *`.
This task wakes the PC at 2:55am — 5 minutes early — and runs the same
watchdog, so by 3am the app is guaranteed to be up for the scheduler to fire.

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Users\capta\source\repos\capnknives\WitsV3-claude\scripts\witsv3_watchdog.ps1"'
$trigger = New-ScheduledTaskTrigger -Daily -At 2:55am
$settings = New-ScheduledTaskSettingsSet -WakeToRun -AllowStartIfOnBattery -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "WitsV3 Wake for Self-Repair" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
```

**Wake timers only work from Sleep, not a full shutdown.** If the PC is
powered off rather than sleeping, Task Scheduler cannot wake it — that needs
a BIOS/UEFI "Power On by RTC Alarm" setting (or Wake-on-LAN) instead, which
is machine-specific and set outside Windows.

Confirm the task is recognized as an allowed wake source:

```powershell
powercfg /waketimers
```

If your power plan has wake timers disabled, enable them: Control Panel →
Power Options → your plan → Change plan settings → Change advanced power
settings → Sleep → Allow wake timers → Enable.

## Removing these tasks

```powershell
Unregister-ScheduledTask -TaskName "WitsV3 Watchdog" -Confirm:$false
Unregister-ScheduledTask -TaskName "WitsV3 Wake for Self-Repair" -Confirm:$false
```

## Logs

The watchdog appends to `logs/watchdog.log` (created on first run) with a
timestamped line every time it runs, whether or not it had to relaunch
anything.
