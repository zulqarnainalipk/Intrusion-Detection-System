param(
  [switch]$NoBrowser,
  [switch]$Rebuild
)

$ErrorActionPreference = "Stop"

$script:Port = 4173
$script:StartedProcessIds = New-Object System.Collections.Generic.List[int]
$script:PidFile = Join-Path $PSScriptRoot ".ids-demo-runtime.json"

function Stop-ProcessTree {
  param(
    [int]$ProcessId
  )

  if ($ProcessId -le 0 -or $ProcessId -eq $PID) {
    return
  }

  if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    taskkill /PID $ProcessId /T /F *> $null
  }
}

function Stop-ExistingRuntime {
  if (Test-Path $script:PidFile) {
    try {
      $savedState = Get-Content $script:PidFile -Raw | ConvertFrom-Json
      foreach ($savedId in @($savedState.serverPid, $savedState.browserPid)) {
        if ($savedId) {
          Stop-ProcessTree -ProcessId ([int]$savedId)
        }
      }
    } catch {
    }

    Remove-Item $script:PidFile -Force -ErrorAction SilentlyContinue
  }

  $listeners = Get-NetTCPConnection -LocalPort $script:Port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique

  foreach ($listenerPid in $listeners) {
    Stop-ProcessTree -ProcessId ([int]$listenerPid)
  }
}

function Wait-ForPort {
  param(
    [int]$Port,
    [int]$TimeoutSeconds = 30
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($connection) {
      return $true
    }

    Start-Sleep -Milliseconds 300
  }

  return $false
}

function Get-BrowserCommand {
  $edgePaths = @(
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
  )
  foreach ($edgePath in $edgePaths) {
    if ($edgePath -and (Test-Path $edgePath)) {
      return @{
        FilePath = $edgePath
        Arguments = @("--app=http://127.0.0.1:$($script:Port)", "--new-window")
      }
    }
  }

  $chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
  )
  foreach ($chromePath in $chromePaths) {
    if ($chromePath -and (Test-Path $chromePath)) {
      return @{
        FilePath = $chromePath
        Arguments = @("--app=http://127.0.0.1:$($script:Port)", "--new-window")
      }
    }
  }

  return $null
}

function Cleanup {
  foreach ($startedId in $script:StartedProcessIds.ToArray()) {
    Stop-ProcessTree -ProcessId $startedId
  }

  Remove-Item $script:PidFile -Force -ErrorAction SilentlyContinue
}

Register-EngineEvent PowerShell.Exiting -Action {
  foreach ($startedId in $script:StartedProcessIds.ToArray()) {
    if ($startedId -and $startedId -ne $PID) {
      taskkill /PID $startedId /T /F *> $null
    }
  }

  Remove-Item $script:PidFile -Force -ErrorAction SilentlyContinue
} | Out-Null

try {
  Set-Location $PSScriptRoot
  Stop-ExistingRuntime

  if (-not (Test-Path (Join-Path $PSScriptRoot "node_modules"))) {
    Write-Host "Installing dependencies..."
    & npm.cmd install
  }

  $distIndexPath = Join-Path $PSScriptRoot "dist\\index.html"
  if ($Rebuild -or -not (Test-Path $distIndexPath)) {
    Write-Host "Building app..."
    & npm.cmd run build
  } else {
    Write-Host "Using existing built app from dist\\."
  }

  $serverProcess = Start-Process -FilePath "node.exe" `
    -ArgumentList @("serve-dist.mjs", "--port", "$script:Port") `
    -WorkingDirectory $PSScriptRoot `
    -PassThru
  $script:StartedProcessIds.Add($serverProcess.Id)

  if (-not (Wait-ForPort -Port $script:Port)) {
    throw "The local server did not start on port $($script:Port)."
  }

  $browserProcess = $null
  if (-not $NoBrowser) {
    $browserCommand = Get-BrowserCommand
    if ($browserCommand) {
      $browserProcess = Start-Process -FilePath $browserCommand.FilePath `
        -ArgumentList $browserCommand.Arguments `
        -WorkingDirectory $PSScriptRoot `
        -PassThru
      $script:StartedProcessIds.Add($browserProcess.Id)
    } else {
      Start-Process "http://127.0.0.1:$($script:Port)"
      Write-Host "Opened the app in your default browser."
    }
  }

  $runtimeState = @{
    serverPid = $serverProcess.Id
    browserPid = if ($browserProcess) { $browserProcess.Id } else { $null }
    port = $script:Port
  }
  $runtimeState | ConvertTo-Json | Set-Content $script:PidFile

  Write-Host ""
  Write-Host "IDS demo is running at http://127.0.0.1:$($script:Port)"
  Write-Host "Close this window to stop the local server."
  if ($browserProcess) {
    Write-Host "Closing the app window will also stop the server."
    Wait-Process -Id $browserProcess.Id
  } else {
    Write-Host "Press Ctrl+C or close this window when you are done."
    while ($true) {
      Start-Sleep -Seconds 2
    }
  }
} finally {
  Cleanup
}
