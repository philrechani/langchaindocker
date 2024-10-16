Set-Location $env:homepathPS

# Get the current date
$currentDate = Get-Date -Format "MM/dd/yyyy"

# Add the date separator to the log file
"=====" | Out-File -FilePath "output.log" -Append
$currentDate | Out-File -FilePath "output.log" -Append
"=====" | Out-File -FilePath "output.log" -Append

# Run docker and get the output
docker compose down | Tee-Object -FilePath "output.log"
docker compose build --no-cache --progress=plain | Tee-Object -FilePath "output.log" -Append
docker compose up -d | Tee-Object -FilePath "output.log" -Append