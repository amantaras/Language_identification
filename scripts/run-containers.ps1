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

Write-Host "All env keys:" -ForegroundColor Yellow
$envTable.Keys | ForEach-Object { Write-Host "  '$_' = '$($envTable[$_])'" -ForegroundColor Yellow }

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

Write-Host "Starting Language Identification (host $lidPort -> 5000)" -ForegroundColor Green
Write-Host "Debug: lidImg='$lidImg'" -ForegroundColor Yellow
$lidImageRef = $lidImg.Trim()
Write-Host "Debug: lidImageRef='$lidImageRef'" -ForegroundColor Yellow
if (-not ($lidImageRef -match ':[^/]+$')) { $lidImageRef = "$lidImageRef:latest" }
Write-Host "Debug: lidImageRef after regex='$lidImageRef'" -ForegroundColor Yellow

$lidCmd = "docker run -d --name $lidName -p $lidPort`:5000 --memory $lidMem --cpus $lidCpu -e Eula=accept -e Billing=$billing -e ApiKey=$key $lidImageRef"
if ($Interactive) {
    $lidCmd = $lidCmd.Replace(" -d ", " ")
}
Write-Host " > $lidCmd" -ForegroundColor DarkGray
Invoke-Expression $lidCmd

Write-Host "Starting EN STT ($enLoc host $enPort -> 5000)" -ForegroundColor Green
$sttImageRef = $sttImg.Trim()
if (-not ($sttImageRef -match ':[^/]+$')) { $sttImageRef = "$sttImageRef:latest" }

$enCmd = "docker run -d --name $enName -p $enPort`:5000 --memory $sttMem --cpus $sttCpu -e Eula=accept -e Billing=$billing -e ApiKey=$key -e SpeechServiceConnection_Locale=$enLoc $sttImageRef"
if ($Interactive) {
    $enCmd = $enCmd.Replace(" -d ", " ")
}
Write-Host " > $enCmd" -ForegroundColor DarkGray
Invoke-Expression $enCmd

Write-Host "Starting AR STT ($arLoc host $arPort -> 5000)" -ForegroundColor Green

$arCmd = "docker run -d --name $arName -p $arPort`:5000 --memory $sttMem --cpus $sttCpu -e Eula=accept -e Billing=$billing -e ApiKey=$key -e SpeechServiceConnection_Locale=$arLoc $sttImageRef"
if ($Interactive) {
    $arCmd = $arCmd.Replace(" -d ", " ")
}
Write-Host " > $arCmd" -ForegroundColor DarkGray
Invoke-Expression $arCmd

Write-Host "All containers started." -ForegroundColor Cyan
if (-not $Interactive) { Write-Host "Use docker logs -f <name> to view logs." -ForegroundColor DarkCyan }
