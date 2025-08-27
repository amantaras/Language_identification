[CmdletBinding()]
param(
    [string[]]$Names = @('speech-lid','speech-stt-en-us','speech-stt-ar-sa')
)
foreach ($n in $Names) {
    if (docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $n }) {
        Write-Host "Stopping $n" -ForegroundColor Yellow
        docker rm -f $n | Out-Null
    }
}
Write-Host "Done." -ForegroundColor Green
