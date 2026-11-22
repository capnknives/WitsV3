# Ensure Docker Desktop is running before WitsV3 starts (sandbox_mode: docker).
# Safe to run repeatedly. Used by start_web_ui.bat and the watchdog.
#
# Critical: resolve docker.exe by full path (many shells lack Docker on PATH)
# and never let `docker info` block forever.

param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [int]$WaitSeconds = 180,
    [int]$InfoTimeoutSeconds = 15,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$logFile = Join-Path $ProjectDir "logs\docker_companion.log"
$script:InfoTimeoutMs = [Math]::Max(1000, $InfoTimeoutSeconds * 1000)

function Write-Log($message) {
    $dir = Split-Path $logFile -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $message"
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Output $line
}

function Get-SandboxMode {
    $configPath = Join-Path $ProjectDir "config.yaml"
    if (-not (Test-Path $configPath)) { return "off" }
    try {
        $match = Select-String -Path $configPath -Pattern '^\s*sandbox_mode:\s*["'']?(\w+)' | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value.ToLowerInvariant() }
    } catch { }
    return "off"
}

function Resolve-DockerExe {
    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    $candidates = @(
        (Join-Path $env:ProgramFiles "Docker\Docker\resources\bin\docker.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\resources\bin\docker.exe")
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    return $null
}

function Ensure-DockerBinOnPath([string]$DockerExe) {
    # Reason: docker-credential-desktop and plugins live next to docker.exe.
    $bin = Split-Path -Parent $DockerExe
    if ($bin -and ($env:PATH -notlike "*$bin*")) {
        $env:PATH = "$bin;$env:PATH"
    }
}

function Test-DockerDaemon {
    param(
        [Parameter(Mandatory = $true)][string]$DockerExe,
        [int]$TimeoutMs = $script:InfoTimeoutMs
    )
    if (-not $DockerExe) { return $false }
    Ensure-DockerBinOnPath $DockerExe

    # Reason: System.Diagnostics.Process + timed WaitForExit - never block the launcher.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $DockerExe
    $psi.Arguments = "info"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $psi.EnvironmentVariables["PATH"] = $env:PATH

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    try {
        [void]$proc.Start()
        # Drain async so a full stdout buffer cannot deadlock WaitForExit.
        $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
        $stderrTask = $proc.StandardError.ReadToEndAsync()
        if (-not $proc.WaitForExit($TimeoutMs)) {
            try { $proc.Kill() } catch { }
            try { [void]$proc.WaitForExit(5000) } catch { }
            Write-Log "docker info timed out after ${TimeoutMs}ms"
            return $false
        }
        [void]$stdoutTask.Wait(2000)
        [void]$stderrTask.Wait(2000)
        $code = $proc.ExitCode
        if ($code -ne 0) {
            $err = ""
            try { $err = $stderrTask.Result } catch { }
            if ($err) {
                $snippet = ($err -split "`n" | Select-Object -First 2) -join " "
                Write-Log "docker info exit=$code ($snippet)"
            } else {
                Write-Log "docker info exit=$code"
            }
            return $false
        }
        return $true
    } catch {
        Write-Log "docker info probe failed: $($_.Exception.Message)"
        return $false
    } finally {
        try { $proc.Dispose() } catch { }
    }
}

$mode = Get-SandboxMode
if (-not $Force -and $mode -ne "docker") {
    Write-Log "Skipping Docker preflight (security.sandbox_mode=$mode)."
    exit 0
}

$dockerExe = Resolve-DockerExe
if (-not $dockerExe) {
    Write-Log "ERROR - docker.exe not found on PATH or under Program Files."
    exit 1
}
Write-Log "Using docker CLI: $dockerExe (info timeout ${script:InfoTimeoutMs}ms)"

if (Test-DockerDaemon -DockerExe $dockerExe) {
    Write-Log "Docker daemon already ready."
    exit 0
}

$desktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $desktop)) {
    $desktop86 = Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktop86) { $desktop = $desktop86 }
}

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
    Write-Log "Waiting for Docker daemon... ${elapsed}s / ${WaitSeconds}s"
    if (Test-DockerDaemon -DockerExe $dockerExe) {
        Write-Log "Docker daemon ready after ${elapsed}s."
        exit 0
    }
}

Write-Log "ERROR - Docker daemon not ready after ${WaitSeconds}s."
exit 1
