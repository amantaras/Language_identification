# Simple test script to debug env loading
$EnvFile = '.\.env'
Write-Host "Loading env file: $EnvFile" -ForegroundColor Cyan

if (-not (Test-Path $EnvFile)) { 
    Write-Host "Env file not found: $EnvFile" -ForegroundColor Red
    return
}

$envTable = @{}
Get-Content $EnvFile | ForEach-Object {
    $l = $_.Trim()
    Write-Host "Line: '$l'" -ForegroundColor Gray
    if (-not $l -or $l.StartsWith('#')) { return }
    $kv = $l -split '=',2
    if ($kv.Count -eq 2) {
        $k = $kv[0].Trim()
        $v = $kv[1].Trim().Trim('"').Trim("'")
        $envTable[$k] = $v
        Write-Host "Added: $k = $v" -ForegroundColor Green
    }
}

Write-Host "Final envTable contents:" -ForegroundColor Cyan
$envTable.Keys | ForEach-Object { 
    Write-Host "  '$_' = '$($envTable[$_])'" -ForegroundColor Cyan 
}

Write-Host "LID_IMAGE = '$($envTable['LID_IMAGE'])'" -ForegroundColor Yellow
Write-Host "STT_IMAGE = '$($envTable['STT_IMAGE'])'" -ForegroundColor Yellow
