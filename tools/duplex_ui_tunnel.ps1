param(
    [ValidateSet("start", "status", "stop")]
    [string]$Action = "status",
    [ValidatePattern("^[A-Za-z0-9._@:-]+$")]
    [string]$HostAlias = "gpu3090",
    [ValidateRange(1024, 65535)]
    [int]$LocalPort = 18000
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tempRoot = Join-Path $repoRoot ".tmp"
$pidPath = Join-Path $tempRoot "duplex_ui_tunnel.pid"
$forwardSpec = "127.0.0.1:$LocalPort" + ":127.0.0.1:8000"

function Test-LocalPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync([System.Net.IPAddress]::Loopback, $Port)
        return $task.Wait(350) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-TunnelProcess {
    if (-not (Test-Path -LiteralPath $pidPath)) { return $null }
    $savedPid = 0
    if (-not [int]::TryParse((Get-Content -Raw -LiteralPath $pidPath), [ref]$savedPid)) { return $null }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $savedPid" -ErrorAction SilentlyContinue
    if ($null -eq $process -or [IO.Path]::GetFileName($process.ExecutablePath) -ne "ssh.exe") { return $null }
    if ($process.CommandLine -notlike "*$forwardSpec*" -or $process.CommandLine -notlike "*$HostAlias*") { return $null }
    return $process
}

switch ($Action) {
    "status" {
        $process = Get-TunnelProcess
        if ($null -ne $process -and (Test-LocalPort $LocalPort)) {
            Write-Output "Tunnel attivo: http://localhost:$LocalPort/ (PID $($process.ProcessId))"
        } else {
            Write-Output "Tunnel non attivo su localhost:$LocalPort."
        }
    }
    "start" {
        $process = Get-TunnelProcess
        if ($null -ne $process -and (Test-LocalPort $LocalPort)) {
            Write-Output "Tunnel già attivo: http://localhost:$LocalPort/ (PID $($process.ProcessId))"
            break
        }
        if (Test-LocalPort $LocalPort) {
            throw "La porta locale $LocalPort è già occupata da un altro processo."
        }
        $ssh = Get-Command ssh.exe -ErrorAction Stop
        New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $stdoutPath = Join-Path $tempRoot "duplex_ui_tunnel_$stamp.out.log"
        $stderrPath = Join-Path $tempRoot "duplex_ui_tunnel_$stamp.err.log"
        $arguments = @(
            "-N", "-L", $forwardSpec,
            "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            $HostAlias
        )
        $startOptions = @{
            FilePath = $ssh.Source
            ArgumentList = $arguments
            WindowStyle = "Hidden"
            PassThru = $true
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
        }
        $started = Start-Process @startOptions
        Start-Sleep -Milliseconds 500
        $stillRunning = Get-Process -Id $started.Id -ErrorAction SilentlyContinue
        if ($null -eq $stillRunning) {
            $detail = if (Test-Path -LiteralPath $stderrPath) { (Get-Content -Raw -LiteralPath $stderrPath).Trim() } else { "SSH non è partito." }
            throw $detail
        }
        Set-Content -LiteralPath $pidPath -Value $started.Id -NoNewline -Encoding ascii
        Write-Output "Tunnel avviato verso $HostAlias. Apri http://localhost:$LocalPort/ dopo che il server remoto è pronto."
        Write-Output "Log SSH: $stderrPath"
    }
    "stop" {
        $process = Get-TunnelProcess
        if ($null -eq $process) {
            if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath }
            Write-Output "Nessun tunnel gestito da questo script risulta attivo."
            break
        }
        Stop-Process -Id $process.ProcessId
        Remove-Item -LiteralPath $pidPath
        Write-Output "Tunnel fermato (PID $($process.ProcessId))."
    }
}
