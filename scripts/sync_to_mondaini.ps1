<#
.SYNOPSIS
  Sincronizza automaticamente il codice locale su Mondaini (RTX 3090).
.DESCRIPTION
  Esclude tassativamente checkpoint, pesi, dataset e file pesanti.
  Supporta modalità One-Shot, Watcher continuo (--watch) e Allineamento Git (--git).
.EXAMPLE
  .\scripts\sync_to_mondaini.ps1 -Watch
  .\scripts\sync_to_mondaini.ps1 -Git
#>
param(
    [switch]$Watch,
    [switch]$Git,
    [string]$File
)

$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

$ScriptPath = Join-Path $PSScriptRoot "sync_to_mondaini.py"

$ArgsList = @()
if ($Watch) { $ArgsList += "--watch" }
if ($Git) { $ArgsList += "--git" }
if ($File) { $ArgsList += @("--file", $File) }

& $VenvPython $ScriptPath @ArgsList
