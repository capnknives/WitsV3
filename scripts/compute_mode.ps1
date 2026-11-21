# Report-only compute check: free/total VRAM and RAM, the ten processes using
# the most RAM, and every process holding the GPU. Closes and changes nothing.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compute_mode.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compute_mode.ps1 -MinFreeVramMb 8000 -MinFreeRamGb 16
#   pwsh -NoProfile -File scripts/compute_mode.ps1
#
# Parameters:
#   -MinFreeVramMb  Free GPU memory threshold in MB (default 6500).
#   -MinFreeRamGb   Free system RAM threshold in GB (default 24).
#
# The last line is "VERDICT: OK" or "VERDICT: SHORT - <what is short>".
# Exit code is 0 in every case. If nvidia-smi is missing, VRAM is reported as unknown.
# With several GPUs, the verdict uses the GPU with the most free memory.

param(
    [int]$MinFreeVramMb = 6500,
    [double]$MinFreeRamGb = 24
)

$ErrorActionPreference = "Continue"
$shortReasons = @()
$freeVramMb = $null
$freeRamGb = $null
$smiPresent = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)

function Write-Section($title) {
    Write-Output ""
    Write-Output "== $title =="
}

# ---- GPU memory ----
Write-Section "GPU memory"
if (-not $smiPresent) {
    Write-Output "nvidia-smi not found on PATH. VRAM check is UNKNOWN."
} else {
    $vramRows = @()
    $raw = & nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0) {
        $vramRows = @($raw | Where-Object { $_ -match '\d' })
    }
    $bestFree = -1
    foreach ($line in $vramRows) {
        $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
        $f = 0
        $t = 0
        if ($parts.Count -ge 2 -and [int]::TryParse($parts[0], [ref]$f) -and [int]::TryParse($parts[1], [ref]$t)) {
            Write-Output "GPU: $f MB free of $t MB total"
            if ($f -gt $bestFree) { $bestFree = $f }
        } else {
            Write-Output "GPU row not readable: $line"
        }
    }
    if ($bestFree -ge 0) { $freeVramMb = $bestFree }
    if ($null -eq $freeVramMb) { Write-Output "VRAM values could not be read. VRAM check is UNKNOWN." }
}

# ---- Processes holding the GPU ----
Write-Section "Processes holding the GPU"
if (-not $smiPresent) {
    Write-Output "Skipped (nvidia-smi not found)."
} else {
    $appRaw = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
    $appRows = @()
    if ($LASTEXITCODE -eq 0) {
        $appRows = @($appRaw | Where-Object { $_ -match '^\s*\d+\s*,' })
    }
    if ($appRows.Count -eq 0) {
        Write-Output "None reported by nvidia-smi."
    }
    foreach ($row in $appRows) {
        $p = @($row -split ',' | ForEach-Object { $_.Trim() })
        $gpuPid = $p[0]
        $used = $p[$p.Count - 1]
        if ($p.Count -ge 3) {
            $name = ($p[1..($p.Count - 2)]) -join ','
        } else {
            $name = ""
        }
        # used_memory reads "[N/A]" on Windows; show it as reported.
        Write-Output ("  PID {0,-8} {1,-40} GPU mem: {2} MB" -f $gpuPid, $name, $used)
    }
}

# ---- System RAM ----
Write-Section "System RAM"
try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    if ($null -eq $os) { throw "no data returned" }
    # Both values are in KB; 1048576 KB = 1 GB.
    $freeRamGb = [math]::Round([double]$os.FreePhysicalMemory / 1048576, 1)
    $totalRamGb = [math]::Round([double]$os.TotalVisibleMemorySize / 1048576, 1)
    Write-Output "Free RAM: $freeRamGb GB of $totalRamGb GB total"
} catch {
    Write-Output "RAM query failed: $($_.Exception.Message). RAM check is UNKNOWN."
}

# ---- Top 10 processes by RAM ----
Write-Section "Top 10 processes by RAM (working set)"
Get-Process -ErrorAction SilentlyContinue |
    Sort-Object WorkingSet64 -Descending |
    Select-Object -First 10 |
    ForEach-Object {
        $mb = [math]::Round($_.WorkingSet64 / 1048576, 0)
        Write-Output ("  {0,-8} {1,-32} {2,8} MB" -f $_.Id, $_.ProcessName, $mb)
    }

# ---- Verdict ----
Write-Section "Verdict"
if ($null -eq $freeVramMb) {
    $shortReasons += "VRAM unknown"
} elseif ($freeVramMb -lt $MinFreeVramMb) {
    $shortReasons += "VRAM free $freeVramMb MB < $MinFreeVramMb MB"
}
if ($null -eq $freeRamGb) {
    $shortReasons += "RAM unknown"
} elseif ($freeRamGb -lt $MinFreeRamGb) {
    $shortReasons += "RAM free $freeRamGb GB < $MinFreeRamGb GB"
}

if ($shortReasons.Count -eq 0) {
    Write-Output "VERDICT: OK"
} else {
    Write-Output ("VERDICT: SHORT - " + ($shortReasons -join "; "))
}
exit 0
