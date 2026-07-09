#Requires -RunAsAdministrator
# Analyze today's BSOD minidump: copy locally, run WinDbg or Python fallback.
$ErrorActionPreference = 'Stop'
$src = 'C:\Windows\Minidump\070826-7703-01.dmp'
$destDir = Join-Path $env:TEMP 'bsod-analysis'
$dest = Join-Path $destDir '070826-7703-01.dmp'
$out = Join-Path $destDir 'analysis.txt'

New-Item -ItemType Directory -Force -Path $destDir | Out-Null

if (-not (Test-Path $src)) {
    Write-Error "Minidump not found: $src"
}

Copy-Item -Force $src $dest
Write-Host "Copied dump to $dest ($((Get-Item $dest).Length) bytes)"

$windbgPaths = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\Debuggers\x64\windbg.exe",
    "$env:ProgramFiles\Windows Kits\10\Debuggers\x64\windbg.exe"
)
$windbg = $windbgPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($windbg) {
    Write-Host "Running WinDbg: $windbg"
    & $windbg -z $dest -c "!analyze -v; lm; q" 2>&1 | Tee-Object -FilePath $out
} else {
    Write-Host "WinDbg not found; using Python minidump parser"
    $py = @'
import struct, sys
from pathlib import Path
from minidump.minidumpfile import MinidumpFile

path = Path(sys.argv[1])
out = Path(sys.argv[2])
lines = [f"File: {path}", f"Size: {path.stat().st_size} bytes", ""]

md = MinidumpFile.parse(str(path))

if md.sysinfo:
    si = md.sysinfo
    lines += [
        "=== System Info ===",
        f"CPU arch: {si.ProcessorArchitecture}",
        f"CPU count: {si.NumberOfProcessors}",
        f"Build: {si.BuildNumber}",
        f"Platform ID: {si.PlatformId}",
        "",
    ]

if md.exception:
    ex = md.exception
    code = ex.ExceptionRecord.ExceptionCode
    addr = ex.ExceptionRecord.ExceptionAddress
    lines += [
        "=== Exception ===",
        f"Code: 0x{code:08X}",
        f"Address: 0x{addr:016X}",
        "",
    ]
    if code == 0x3B:
        lines.append("Bugcheck 0x3B = SYSTEM_SERVICE_EXCEPTION")
    if code == 0xC0000005:
        lines.append("Subcode 0xC0000005 = ACCESS_VIOLATION")

if md.modules:
    lines += ["", "=== Loaded Modules (kernel-relevant) ==="]
    for m in sorted(md.modules.modules, key=lambda x: x.baseaddress):
        name = m.name or "?"
        if any(k in name.lower() for k in ("ntoskrnl", "nvlddmkm", "dxgkrnl", "win32k", "tcpip", "fltmgr", "storport", "iaStor", "amd", "nv", "rz", "synapse")):
            lines.append(f"  0x{m.baseaddress:016X}  {name}")

if md.misc_info and hasattr(md.misc_info, "info") and md.misc_info.info:
    mi = md.misc_info.info
    if hasattr(mi, "KernelDebuggerDataBlockRva") and mi.KernelDebuggerDataBlockRva:
        lines += ["", f"KDBG RVA: 0x{mi.KernelDebuggerDataBlockRva:X}"]

out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
'@
    $pyPath = Join-Path $destDir 'parse_dump.py'
    Set-Content -Path $pyPath -Value $py -Encoding UTF8
    python $pyPath $dest $out
}

Write-Host ""
Write-Host "=== Analysis saved to: $out ==="
