$ErrorActionPreference = "Stop"

# Activate local venv
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning ".venv not found. Create it first: python -m venv .venv"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = "runs\full_generation_$timestamp.log"
$null = New-Item -ItemType Directory -Path runs -Force
Start-Transcript -Path $logPath -Append | Out-Null

try {
    # Pass classes as an array so each becomes a separate CLI arg
    $classes = @('bird_drop','clean','dusty','electrical_damage','physical_damage','snow_covered')
    $outRoot = "generated"

    Write-Host "[1/3] Structural Consistency..." -ForegroundColor Cyan
    python image_gen\orchestrate_generation.py --classes $classes --ratio 100,0,0 --output-root $outRoot --manifest runs\manifest_generated_sc_$timestamp.json --class-count bird_drop=180 --class-count clean=185 --class-count dusty=190 --class-count electrical_damage=246 --class-count physical_damage=260 --class-count snow_covered=231
    if ($LASTEXITCODE -ne 0) { throw "SC phase failed with exit code $LASTEXITCODE" }

    Write-Host "[2/3] Domain Adaptation..." -ForegroundColor Cyan
    python image_gen\orchestrate_generation.py --classes $classes --ratio 0,100,0 --output-root $outRoot --manifest runs\manifest_generated_da_$timestamp.json --class-count bird_drop=40 --class-count clean=41 --class-count dusty=42 --class-count electrical_damage=55 --class-count physical_damage=58 --class-count snow_covered=51
    if ($LASTEXITCODE -ne 0) { throw "DA phase failed with exit code $LASTEXITCODE" }

    Write-Host "[3/3] Text-to-Image..." -ForegroundColor Cyan
    python image_gen\orchestrate_generation.py --classes $classes --ratio 0,0,100 --output-root $outRoot --manifest runs\manifest_generated_t2i_$timestamp.json --class-count bird_drop=79 --class-count clean=83 --class-count dusty=86 --class-count electrical_damage=109 --class-count physical_damage=116 --class-count snow_covered=104
    if ($LASTEXITCODE -ne 0) { throw "T2I phase failed with exit code $LASTEXITCODE" }

    Write-Host "All phases completed successfully." -ForegroundColor Green
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}


