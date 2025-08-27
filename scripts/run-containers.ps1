<#
Simplified, sanitized launcher script for Azure Speech containers.
Rewritten to avoid any potential hidden characters / parsing issues.

Usage:
    pwsh ./scripts/run-containers.ps1 [-EnvFile ./.env] [-Interactive] [-ForceRecreate]
#>
[CmdletBinding()]
param(
        [string]$EnvFile = (Join-Path $PSScriptRoot '..' '.env'),
        [switch]$Interactive,
        [switch]$ForceRecreate
)

$ErrorActionPreference = 'Stop'

Write-Host "Loading env file: $EnvFile" -ForegroundColor Cyan
if (-not (Test-Path $EnvFile)) { throw "Env file not found: $EnvFile" }

# --- Parse .env (no export to $Env:, just local hashtable) ---
$envTable = @{}
Get-Content $EnvFile | ForEach-Object {
        $l = $_.Trim()
        if (-not $l -or $l.StartsWith('#')) { return }
        $kv = $l -split '=',2
        if ($kv.Count -eq 2) {
                $k = $kv[0].Trim()
                $v = $kv[1].Trim().Trim('"').Trim("'")
                $envTable[$k] = $v
        }
}

# --- Required keys (comma-free list to avoid any hidden char issues) ---
$required = @(
        'SPEECH_BILLING_ENDPOINT'
        'SPEECH_API_KEY'
        'LID_PORT'
        'EN_PORT'
        'AR_PORT'
        'LID_IMAGE'
        'STT_IMAGE'
        'EN_LOCALE'
        'AR_LOCALE'
)
$missing = @()
foreach ($r in $required) {
                $present = $false
                if ($envTable.ContainsKey($r)) { $val = $envTable[$r]; if (-not [string]::IsNullOrWhiteSpace($val)) { $present = $true } }
                if (-not $present) { $missing += $r }
}
if ($missing.Count -gt 0) {
                $msg = "Missing required variables in .env: " + ($missing -join ', ')
                throw $msg
}

# Defaults
if (-not $envTable['LID_MEMORY']) { $envTable['LID_MEMORY'] = '1g' }
if (-not $envTable['LID_CPUS'])   { $envTable['LID_CPUS']   = '1' }
if (-not $envTable['STT_MEMORY']) { $envTable['STT_MEMORY'] = '2g' }
if (-not $envTable['STT_CPUS'])   { $envTable['STT_CPUS']   = '2' }

Write-Host "Env summary:" -ForegroundColor DarkGray
foreach ($k in $required) {
        $val = if ($k -match 'KEY') { '***REDACTED***' } else { $envTable[$k] }
        Write-Host "  $k = $val" -ForegroundColor DarkGray
}

# Local vars
$billing = $envTable['SPEECH_BILLING_ENDPOINT']
$key     = $envTable['SPEECH_API_KEY']
$lidPort = $envTable['LID_PORT']
$enPort  = $envTable['EN_PORT']
$arPort  = $envTable['AR_PORT']
$lidImg  = $envTable['LID_IMAGE']
$sttImg  = $envTable['STT_IMAGE']
$enLoc   = $envTable['EN_LOCALE']
$arLoc   = $envTable['AR_LOCALE']
$lidMem  = $envTable['LID_MEMORY']
$lidCpu  = $envTable['LID_CPUS']
$sttMem  = $envTable['STT_MEMORY']
$sttCpu  = $envTable['STT_CPUS']

$lidName = 'speech-lid'
$enName  = "speech-stt-$($enLoc.ToLower())"
$arName  = "speech-stt-$($arLoc.ToLower())"

$detach = if ($Interactive) { @() } else { @('-d') }

function Remove-IfExists([string]$n) {
        $exists = docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $n }
        if ($exists) { docker rm -f $n | Out-Null }
}

if ($ForceRecreate) {
        Write-Host "Force removing any existing containers..." -ForegroundColor Yellow
        Remove-IfExists $lidName
        Remove-IfExists $enName
        Remove-IfExists $arName
}

Write-Host "Starting Language Identification ($lidPort)" -ForegroundColor Green
$lidImageRef = if ($lidImg -match ':[^/]+$') { $lidImg } else { "$lidImg:latest" }
if (-not $lidImageRef) { throw "LID image reference resolved empty (source '$lidImg')" }
Write-Host " Image: $lidImageRef" -ForegroundColor DarkGray
docker run @detach --name $lidName -p "${lidPort}:5000" --memory $lidMem --cpus $lidCpu `
        $lidImageRef `
        Eula=accept Billing=$billing ApiKey=$key

Write-Host "Starting EN STT ($enLoc on $enPort)" -ForegroundColor Green
$sttImageRef = if ($sttImg -match ':[^/]+$') { $sttImg } else { "$sttImg:latest" }
if (-not $sttImageRef) { throw "STT image reference resolved empty (source '$sttImg')" }
Write-Host " Image: $sttImageRef" -ForegroundColor DarkGray
docker run @detach --name $enName -p "${enPort}:5000" --memory $sttMem --cpus $sttCpu `
        $sttImageRef `
        Eula=accept Billing=$billing ApiKey=$key SpeechServiceConnection_Locale=$enLoc

Write-Host "Starting AR STT ($arLoc on $arPort)" -ForegroundColor Green
Write-Host " Image: $sttImageRef" -ForegroundColor DarkGray
docker run @detach --name $arName -p "${arPort}:5000" --memory $sttMem --cpus $sttCpu `
        $sttImageRef `
        Eula=accept Billing=$billing ApiKey=$key SpeechServiceConnection_Locale=$arLoc

Write-Host "All containers started." -ForegroundColor Cyan
if (-not $Interactive) { Write-Host "Use docker logs -f <name> to view logs." -ForegroundColor DarkCyan }
