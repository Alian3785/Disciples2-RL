param(
    [Parameter(Mandatory = $true)][string]$Run,
    [Parameter(Mandatory = $true)][int]$Threshold,
    [string]$OutputDir = "outputs\hparam_sweep_1m",
    [string]$ErrLogName = "master_resume_03.err.log",
    [int]$TimeoutMinutes = 25
)

$log = Join-Path $OutputDir "logs\$($Run)_train.log"
$summary = Join-Path $OutputDir "summaries\$($Run).txt"
$evalLog = Join-Path $OutputDir "logs\$($Run)_eval.log"
$err = Join-Path $OutputDir $ErrLogName
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

function Last-Match([string]$Text, [string]$Pattern) {
    $matches = [regex]::Matches($Text, $Pattern)
    if ($matches.Count -gt 0) {
        return $matches[$matches.Count - 1].Groups[1].Value
    }
    return $null
}

while ((Get-Date) -lt $deadline) {
    $summaryIsCurrent = $false
    if (Test-Path $summary) {
        $summaryTime = (Get-Item $summary).LastWriteTime
        $logTime = if (Test-Path $log) { (Get-Item $log).LastWriteTime } else { [datetime]::MinValue }
        $summaryIsCurrent = $summaryTime -ge $logTime
    }

    if ($summaryIsCurrent) {
        $evalText = if (Test-Path $evalLog) { Get-Content $evalLog -Raw } else { "" }
        $wins = Last-Match $evalText 'wins:\s*(\d+)'
        $defeats = Last-Match $evalText 'defeats:\s*(\d+)'
        $timeouts = Last-Match $evalText 'timeouts:\s*(\d+)'
        $reward = Last-Match $evalText 'mean_reward:\s*([-+0-9.eE]+)'
        $enemies = Last-Match $evalText 'mean_enemies_defeated:\s*([-+0-9.eE]+)'
        "RUN_DONE $Run wins=$wins defeats=$defeats timeouts=$timeouts mean_reward=$reward mean_enemies=$enemies"
        Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        exit 0
    }

    if ((Test-Path $err) -and ((Get-Item $err).Length -gt 0)) {
        "ERROR_LOG_NONEMPTY $err"
        Get-Content $err -Tail 120
        exit 1
    }

    if (Test-Path $log) {
        $text = Get-Content $log -Raw
        $steps = Last-Match $text 'total_timesteps\s+\|\s+([0-9]+)'
        if ($steps -and ([int]$steps -ge $Threshold)) {
            $reward = Last-Match $text 'ep_rew_mean\s+\|\s+([-+0-9.eE]+)'
            $length = Last-Match $text 'ep_len_mean\s+\|\s+([-+0-9.eE]+)'
            "RUN_PROGRESS $Run steps=$steps ep_rew_mean=$reward ep_len_mean=$length"
            Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            exit 0
        }
    }

    Start-Sleep -Seconds 30
}

"TIMEOUT_WAITING $Run threshold=$Threshold"
Get-Date -Format "yyyy-MM-dd HH:mm:ss"
exit 2
