# Restore wits_memory.json + FAISS index from the personal WitsV3 runtime worktree.
# Run from repo root AFTER stopping run_web.py / smoke harnesses that hold file locks.
#
#   powershell -File scripts/restore_runtime_memory.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtime = "c:\Users\capta\source\repos\capnknives\WitsV3\var\data"
$target = Join-Path $repoRoot "var\data"
$ts = Get-Date -Format "yyyyMMdd-HHmmss"

foreach ($name in @("wits_memory.json", "wits_faiss_index.bin")) {
    $src = Join-Path $runtime $name
    $dst = Join-Path $target $name
    if (-not (Test-Path $src)) {
        Write-Error "Missing source: $src"
    }
    if (Test-Path $dst) {
        Copy-Item $dst "$dst.bak-$ts"
    }
    Copy-Item $src $dst -Force
    Write-Host "Restored $name ($((Get-Item $dst).Length) bytes)"
}

Set-Location $repoRoot
python scripts/analyze_memory.py
