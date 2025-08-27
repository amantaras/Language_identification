<#
Clean Azure Speech containers launcher script.
Usage: pwsh ./scripts/run-containers-simple.ps1 [-Interactive] [-ForceRecreate]
#>
[CmdletBinding()]
param(
    [switch]$Interactive,
    [switch]$ForceRecreate
)

$ErrorActionPreference = 'Stop'

# Load .env file
$EnvFile = '.\.env'
Write-Host "Loading env file: $EnvFile" -ForegroundColor Cyan

if (-not (Test-Path $EnvFile)) { 
    throw "Env file not found: $EnvFile"
}

$envTable = @{}
Get-Content $EnvFile | ForEach-Object {
    $l = $_.Trim()
    if (-not $l -or $l.StartsWith('#')) { return }
    $kv = $l -split '=', 2
    if ($kv.Count -eq 2) {
        $k = $kv[0].Trim()
        $v = $kv[1].Trim().Trim('"').Trim("'")
        $envTable[$k] = $v
    }
}

# Extract variables
$billing = $envTable['SPEECH_BILLING_ENDPOINT']
$key = $envTable['SPEECH_API_KEY']
$lidPort = $envTable['LID_PORT']
$enPort = $envTable['EN_PORT']
$arPort = $envTable['AR_PORT']
$lidImage = $envTable['LID_IMAGE'] + ':latest'
$sttImage = $envTable['STT_IMAGE'] + ':latest'
$enLocale = $envTable['EN_LOCALE']
$arLocale = $envTable['AR_LOCALE']

Write-Host "Configuration loaded:" -ForegroundColor Green
Write-Host "  Billing: $billing" -ForegroundColor DarkGray
Write-Host "  LID Port: $lidPort" -ForegroundColor DarkGray
Write-Host "  EN Port: $enPort" -ForegroundColor DarkGray
Write-Host "  AR Port: $arPort" -ForegroundColor DarkGray
Write-Host "  LID Image: $lidImage" -ForegroundColor DarkGray
Write-Host "  STT Image: $sttImage" -ForegroundColor DarkGray

# Container names
$lidName = 'speech-lid'
$enName = 'speech-stt-en-us'
$arName = 'speech-stt-ar-sa'

# Remove existing containers if requested
if ($ForceRecreate) {
    Write-Host "Removing existing containers..." -ForegroundColor Yellow
    @($lidName, $enName, $arName) | ForEach-Object {
        $existing = docker ps -a -q --filter "name=$_"
        if ($existing) {
            Write-Host "  Removing $_" -ForegroundColor DarkYellow
            docker rm -f $_
        }
    }
}

# Common args
$detachFlag = if ($Interactive) { '' } else { '-d' }

function Start-ContainerIfMissing {
    param(
        [string]$Name,
        [string[]]$DockerArgs
    )
    $exists = docker ps -a -q --filter "name=$Name"
    if ($exists) {
        $running = docker ps -q --filter "name=$Name"
        if ($running) {
            Write-Host "Container '$Name' already running; skipping." -ForegroundColor Yellow
        } else {
            Write-Host "Container '$Name' exists but not running; starting..." -ForegroundColor Yellow
            docker start $Name | Out-Null
        }
    } else {
        Write-Host "Starting container '$Name'..." -ForegroundColor Green
        Write-Host "  docker $($DockerArgs -join ' ')" -ForegroundColor DarkGray
        & docker @DockerArgs
    }
}

# Start Language Identification container (idempotent)
$lidArgs = @(
    'run', $detachFlag, '--name', $lidName,
    '-p', "$lidPort`:$lidPort",  # Map to the same port inside the container
    '--memory', '1g', '--cpus', '1',
    '-e', 'Eula=accept',
    '-e', "Billing=$billing",
    '-e', "ApiKey=$key",
    $lidImage
) | Where-Object { $_ -ne '' }
Start-ContainerIfMissing -Name $lidName -DockerArgs $lidArgs

# Start EN STT container (idempotent)
$enArgs = @(
    'run', $detachFlag, '--name', $enName,
    '-p', "$enPort`:$enPort",  # Map to the same port inside the container
    '--memory', '4g', '--cpus', '4',
    '-e', 'Eula=accept',
    '-e', "Billing=$billing",
    '-e', "ApiKey=$key",
    '-e', "SpeechServiceConnection_Locale=$enLocale",
    $sttImage
) | Where-Object { $_ -ne '' }
Start-ContainerIfMissing -Name $enName -DockerArgs $enArgs

# Start AR STT container (idempotent)
$arArgs = @(
    'run', $detachFlag, '--name', $arName,
    '-p', "$arPort`:$arPort",  # Map to the same port inside the container
    '--memory', '4g', '--cpus', '4',
    '-e', 'Eula=accept',
    '-e', "Billing=$billing",
    '-e', "ApiKey=$key",
    '-e', "SpeechServiceConnection_Locale=$arLocale",
    $sttImage
) | Where-Object { $_ -ne '' }
Start-ContainerIfMissing -Name $arName -DockerArgs $arArgs

Write-Host "All containers started." -ForegroundColor Cyan
if (-not $Interactive) { 
    Write-Host "Use 'docker logs -f <container-name>' to view logs." -ForegroundColor DarkCyan 
    Write-Host "Use 'docker ps' to check container status." -ForegroundColor DarkCyan
}
