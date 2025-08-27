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
$billing = ($envTable['SPEECH_BILLING_ENDPOINT'] | ForEach-Object { $_.Trim() })
$key     = ($envTable['SPEECH_API_KEY'] | ForEach-Object { $_.Trim() })
$lidPort = ($envTable['LID_PORT'] | ForEach-Object { $_.Trim() })
$enPort  = ($envTable['EN_PORT'] | ForEach-Object { $_.Trim() })
$arPort  = ($envTable['AR_PORT'] | ForEach-Object { $_.Trim() })
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

Write-Host "Starting Language Identification (host port $lidPort -> container 5000)" -ForegroundColor Green
$lidImageRef = if ($lidImg -match ':[^/]+$') { $lidImg } else { "$lidImg:latest" }
if (-not $lidPort) { throw "LID_PORT empty" }
Write-Host " > docker run -d --name $lidName -p $lidPort:5000 $lidImageRef Eula=accept Billing=<endpoint> ApiKey=<key>" -ForegroundColor DarkGray
docker run @detach --name $lidName -p "$lidPort:5000" $lidImageRef Eula=accept Billing=$billing ApiKey=$key

Write-Host "Starting EN STT ($enLoc host port $enPort -> 5000)" -ForegroundColor Green
$sttImageRef = if ($sttImg -match ':[^/]+$') { $sttImg } else { "$sttImg:latest" }
if (-not $enPort) { throw "EN_PORT empty" }
Write-Host " > docker run -d --name $enName -p $enPort:5000 $sttImageRef Eula=accept Billing=<endpoint> ApiKey=<key> SpeechServiceConnection_Locale=$enLoc" -ForegroundColor DarkGray
docker run @detach --name $enName -p "$enPort:5000" $sttImageRef Eula=accept Billing=$billing ApiKey=$key SpeechServiceConnection_Locale=$enLoc

Write-Host "Starting AR STT ($arLoc host port $arPort -> 5000)" -ForegroundColor Green
if (-not $arPort) { throw "AR_PORT empty" }
Write-Host " > docker run -d --name $arName -p $arPort:5000 $sttImageRef Eula=accept Billing=<endpoint> ApiKey=<key> SpeechServiceConnection_Locale=$arLoc" -ForegroundColor DarkGray
docker run @detach --name $arName -p "$arPort:5000" $sttImageRef Eula=accept Billing=$billing ApiKey=$key SpeechServiceConnection_Locale=$arLoc

Write-Host "All containers started." -ForegroundColor Cyan
if (-not $Interactive) { Write-Host "Use docker logs -f <name> to view logs." -ForegroundColor DarkCyan }
