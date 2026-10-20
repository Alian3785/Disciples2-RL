param(
    [string]$Output = ('results/gpu-tests/' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [switch]$ColdCache
)

# Start before launching WSL: its startup belongs to the ten-minute budget.
$testStartedAt = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$projectRoot = Split-Path -Parent $PSScriptRoot
$testArguments = @('scripts/test_gpu.py', '--output', $Output,
    '--wall-started-at', $testStartedAt.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($ColdCache) { $testArguments += '--cold-cache' }
& wsl.exe -d Ubuntu-24.04 -u minigrid --cd $projectRoot -- bash scripts/run_gpu.sh @testArguments
exit $LASTEXITCODE
