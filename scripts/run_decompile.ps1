# run_decompile.ps1
# Reads IDA path from settings.json, sets env var, and runs batch_decompile_ida.py.
$settings = Get-Content "$PSScriptRoot\..\settings.json" -Raw | ConvertFrom-Json
$env:IDA_PATH = $settings.ida_path
& "$PSScriptRoot\..\venv\Scripts\python.exe" "$PSScriptRoot\..\batch_decompile_ida.py"
