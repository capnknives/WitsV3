#Requires -RunAsAdministrator
$report = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportQueue' -Filter 'Kernel_3b_*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $report) {
    Write-Output 'No Kernel_3b WER report found in ReportQueue'
    exit 1
}
Write-Output "Report: $($report.FullName)"
Get-ChildItem $report.FullName -Recurse -Force | Select-Object FullName, Length, LastWriteTime
Write-Output '--- Report.wer ---'
Get-Content (Join-Path $report.FullName 'Report.wer') -ErrorAction SilentlyContinue
Write-Output '--- WERInternalMetadata.xml ---'
Get-Content (Join-Path $report.FullName 'WERInternalMetadata.xml') -ErrorAction SilentlyContinue
