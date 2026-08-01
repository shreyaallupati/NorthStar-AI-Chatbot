# Start North Star Support Bot (backend + frontend) with one command.
# Usage:  .\run.ps1        starts both servers
#         .\run.ps1 -Smoke runs the smoke tests and exits
param([switch]$Smoke)

# Native tools here (py, pip, npm, uvicorn) write progress and warnings to
# stderr, which "Stop" would turn into terminating errors. Failures are caught
# explicitly via $LASTEXITCODE and Die instead.
$ErrorActionPreference = "Continue"

$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Die($msg) { Write-Host "Error: $msg" -ForegroundColor Red; exit 1 }

function Get-VenvPython($dir) {
    $win = Join-Path $dir "Scripts\python.exe"
    $nix = Join-Path $dir "bin/python"
    if (Test-Path $win) { return $win }
    if (Test-Path $nix) { return $nix }
    return $null
}

# A venv is only usable if pip works; the repo may contain a broken 3.14 venv.
function Test-VenvOk($dir) {
    $py = Get-VenvPython $dir
    if (-not $py) { return $false }
    & $py -m pip --version 2>&1 | Out-Null
    return $LASTEXITCODE -eq 0
}

# Prefer Python 3.10-3.12; 3.13+ lacks prebuilt wheels for our pinned deps.
# Returns an object so a single-element result is not collapsed into a string.
function Find-Python {
    # The py launcher only reports versions it has registered; probing an
    # unregistered version prints to stderr, so swallow both streams.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10")) {
            & py "-$v" --version 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return [pscustomobject]@{ Exe = "py"; PreArgs = @("-$v") }
            }
        }
    }
    # An interpreter can be installed yet absent from the launcher registry.
    foreach ($base in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles\Python", "C:\")) {
        foreach ($v in @("312", "311", "310")) {
            $cand = Join-Path $base "Python$v\python.exe"
            if (Test-Path $cand) {
                return [pscustomobject]@{ Exe = $cand; PreArgs = @() }
            }
        }
    }
    return $null
}

# --- backend venv ---------------------------------------------------------
$Venv = $null
foreach ($cand in @((Join-Path $Backend ".venv311"), (Join-Path $Backend ".venv"))) {
    if ((Test-Path $cand) -and (Test-VenvOk $cand)) { $Venv = $cand; break }
}

if (-not $Venv) {
    $pyCmd = Find-Python
    if (-not $pyCmd) { Die "No Python 3.10-3.12 found. Install one, then re-run." }
    $Venv = Join-Path $Backend ".venv311"
    if (Test-Path $Venv) { $Venv = Join-Path $Backend ".venv-run" }
    Say "Creating virtualenv at $(Split-Path $Venv -Leaf) using $($pyCmd.Exe) $($pyCmd.PreArgs -join ' ')"
    $venvArgs = @($pyCmd.PreArgs) + @("-m", "venv", $Venv)
    & $pyCmd.Exe @venvArgs
    if (-not (Get-VenvPython $Venv)) { Die "Failed to create virtualenv." }
    $freshVenv = $true
}

$Py = Get-VenvPython $Venv
if (-not $Py) { Die "Virtualenv at $Venv looks broken. Delete it and re-run." }

if ($freshVenv) { & $Py -m pip install --quiet --upgrade pip }

# Always reconcile against requirements.txt so newly added packages are picked
# up in an existing virtualenv. This is a no-op once satisfied.
Say "Checking backend dependencies"
& $Py -m pip install --quiet --disable-pip-version-check -r (Join-Path $Backend "requirements.txt")
if ($LASTEXITCODE -ne 0) { Die "Dependency install failed." }

if (-not (Test-Path (Join-Path $Backend "app\data\northstar.db"))) {
    Say "Seeding mock data"
    Push-Location $Backend; & $Py seed_mock_data.py; Pop-Location
}

if ($Smoke) {
    Say "Running smoke tests"
    Push-Location $Backend; & $Py smoke_test.py; $code = $LASTEXITCODE; Pop-Location
    exit $code
}

# --- pick a free backend port --------------------------------------------
$Port = $null
foreach ($p in @(8000, 8080, 8001, 8090)) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
        $listener.Start()
        $Port = $p
    } catch {
        continue
    } finally {
        if ($listener) { $listener.Stop() }
    }
    if ($Port) { break }
}
if (-not $Port) { Die "No free port among 8000/8080/8001/8090." }

# Keep the frontend pointed at whichever port we landed on.
"VITE_API_URL=http://127.0.0.1:$Port" | Set-Content (Join-Path $Frontend ".env") -Encoding ascii

# --- frontend deps -------------------------------------------------------
if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Say "Installing frontend dependencies (first run, ~1-3 min)"
    Push-Location $Frontend; npm install; Pop-Location
}

# --- start both ----------------------------------------------------------
Say "Starting backend on http://127.0.0.1:$Port"
$backProc = Start-Process -FilePath $Py `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $Backend -NoNewWindow -PassThru

# Wait for the API to answer before booting the UI.
$healthy = $false
foreach ($i in 1..40) {
    if ($backProc.HasExited) { Die "Backend exited during startup." }
    try {
        Invoke-WebRequest "http://127.0.0.1:$Port/health" -TimeoutSec 2 -UseBasicParsing | Out-Null
        $healthy = $true
        break
    } catch { Start-Sleep -Milliseconds 500 }
}
if ($healthy) { Say "Backend healthy" } else { Say "Backend slow to respond; continuing" }

Say "Starting frontend on http://127.0.0.1:5173"
$frontProc = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "5173") `
    -WorkingDirectory $Frontend -NoNewWindow -PassThru

Write-Host ""
Write-Host "  North Star Support Bot is running." -ForegroundColor Green
Write-Host ""
Write-Host "    Chat UI   http://127.0.0.1:5173"
Write-Host "    API docs  http://127.0.0.1:$Port/docs"
Write-Host "    Health    http://127.0.0.1:$Port/health"
Write-Host ""
Write-Host "  Press Ctrl+C to stop both servers."
Write-Host ""

try {
    while (-not $backProc.HasExited -and -not $frontProc.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    Say "Shutting down"
    foreach ($p in @($backProc, $frontProc)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
