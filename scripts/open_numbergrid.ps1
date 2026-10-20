param([string]$Model = '', [string]$Map = '')
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
$viewerUrl = 'http://127.0.0.1:8769/viewer.html'
$isReady = $false
$health = $null
try { $health = Invoke-RestMethod 'http://127.0.0.1:8769/api/health' -TimeoutSec 2; $isReady = $health.ok } catch {}
if ($isReady -and $Model -and -not $health.agent) {
    throw 'Сервер NumberGrid уже запущен без модели. Остановите его и повторите запуск с моделью.'
}
if ($isReady -and $Map -and $health.map -ne $Map) {
    throw "Сервер NumberGrid уже запущен с картой $($health.map). Остановите его и повторите запуск с картой $Map."
}
if (-not $isReady) {
    $logDir = Join-Path $repoPath 'results'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $wslArgs = @('-d', 'Ubuntu-24.04', '-u', 'minigrid', '--cd', ('"' + $repoPath + '"'), '--',
                 'bash', 'scripts/run_gpu.sh', 'serve_number_grid.py', '--port', '8769')
    if ($Model) { $wslArgs += @('--model', ('"' + $Model + '"')) }
    if ($Map) { $wslArgs += @('--map', ('"' + $Map + '"')) }
    Start-Process -FilePath 'wsl.exe' -ArgumentList $wslArgs -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'human-mode.log') `
        -RedirectStandardError (Join-Path $logDir 'human-mode-errors.log')
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 500
        try { $isReady = (Invoke-RestMethod 'http://127.0.0.1:8769/api/health' -TimeoutSec 2).ok } catch {}
        if ($isReady) { break }
    }
}
if (-not $isReady) { throw 'Не удалось запустить NumberGrid. Проверьте results/human-mode-errors.log.' }
Start-Process $viewerUrl
