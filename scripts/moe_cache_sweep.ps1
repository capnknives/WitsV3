<#
.SYNOPSIS
  Sweep llama-server's MoE expert-cache size and benchmark each setting.

.DESCRIPTION
  For each value in -CacheMib this script starts llama-server on -ModelPath, waits for /health,
  runs scripts\bench_models.py --backend openai against it, stops the server, and sleeps for a
  cool-down. A failure at one size is reported and the sweep continues with the next size.
  A size of 0 means the baseline with no cache flags. --no-kv-offload is never passed
  (reported broken together with the cache upstream).

  Each server's stderr goes to a log file under $env:TEMP; the path is printed.
  Results are appended to -Out by bench_models.py, labelled with the cache size and share.

  This script downloads nothing and changes no settings: it only launches and stops the
  llama-server process it started.

.EXAMPLE
  .\scripts\moe_cache_sweep.ps1 -ModelPath D:\OllamaModels\blobs\sha256-abc123
#>
param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$LlamaServerPath = "D:\DevData\llama.cpp\b12359\llama-server.exe",
    [int[]]$CacheMib = @(0, 1728, 2560, 3072),
    [double]$CpuShare = 0.90,
    [int]$Ctx = 8192,
    [int]$Threads = 6,
    [int]$Port = 8080,
    [int]$CooldownSeconds = 45,
    [string]$Python = "python",
    [string]$Out = "docs\benchmarks\results.md"
)

$ErrorActionPreference = "Stop"
$BenchScript = Join-Path $PSScriptRoot "bench_models.py"
$BaseUrl = "http://127.0.0.1:$Port"
$Share = $CpuShare.ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture)

if (-not (Test-Path -LiteralPath $ModelPath)) { throw "Model file not found: $ModelPath" }
if (-not (Test-Path -LiteralPath $LlamaServerPath)) { throw "llama-server not found: $LlamaServerPath" }

foreach ($size in $CacheMib) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $errLog = Join-Path $env:TEMP "llama-server-cache$size-$stamp.err.log"
    $outLog = Join-Path $env:TEMP "llama-server-cache$size-$stamp.out.log"
    $label = "moe-cache-mib=$size cpu-share=$Share"
    if ($size -le 0) { $label = "moe-cache off (baseline)" }
    Write-Host "=== $label ==="
    Write-Host "Server stderr log: $errLog"

    $serverArgs = @(
        "-m", $ModelPath, "-ngl", "99", "--cpu-moe", "--fit", "off", "-fa", "on",
        "-c", "$Ctx", "-np", "1", "-t", "$Threads", "--host", "127.0.0.1", "--port", "$Port"
    )
    if ($size -gt 0) {
        $serverArgs += @("--moe-cache-mib", "$size", "--moe-cache-cpu-share", $Share)
    }

    $proc = $null
    try {
        $proc = Start-Process -FilePath $LlamaServerPath -ArgumentList $serverArgs `
            -RedirectStandardError $errLog -RedirectStandardOutput $outLog `
            -PassThru -WindowStyle Hidden

        $ready = $false
        $deadline = (Get-Date).AddMinutes(3)
        while ((Get-Date) -lt $deadline) {
            if ($proc.HasExited) { throw "llama-server exited early (code $($proc.ExitCode)); see $errLog" }
            try {
                $health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 5
                if ($health.status -eq "ok") { $ready = $true; break }
            } catch {
                # Still loading (503) or not listening yet; keep polling.
            }
            Start-Sleep -Seconds 2
        }
        if (-not $ready) { throw "llama-server not healthy within 3 minutes; see $errLog" }

        $modelName = Split-Path -Leaf $ModelPath
        & $Python $BenchScript --backend openai --base-url $BaseUrl --models $modelName `
            --runs 2 --ctx "$Ctx" --label $label --out $Out
        if ($LASTEXITCODE -ne 0) { throw "bench_models.py exited with code $LASTEXITCODE" }
    } catch {
        Write-Warning "Cache size $size failed: $($_.Exception.Message)"
    } finally {
        if ($null -ne $proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force
            $proc.WaitForExit()
        }
    }

    Write-Host "Cooling down for $CooldownSeconds s"
    Start-Sleep -Seconds $CooldownSeconds
}
Write-Host "Done. Results appended to $Out"
