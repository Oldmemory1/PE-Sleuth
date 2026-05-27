# clean.ps1
# Clean all intermediate outputs and test artifacts.
$ErrorActionPreference = "SilentlyContinue"

Remove-Item "$PSScriptRoot\..\Decompile\Decompile_Input_Raw_PE\toolkit.exe" -Force
Remove-Item "$PSScriptRoot\..\Decompile\Decompile_Output_C_Code\toolkit.c" -Force
Remove-Item "$PSScriptRoot\..\Classify\Classify_Input_C_Code\toolkit.c" -Force

Get-ChildItem "$PSScriptRoot\..\Classify\Classify_Output_All" -Directory | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
}

Remove-Item "$PSScriptRoot\..\reports\toolkit" -Recurse -Force

Write-Output "Clean done."
