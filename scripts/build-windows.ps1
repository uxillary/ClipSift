[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$expectedVenv = Join-Path $repoRoot ".venv"
$python = Join-Path $expectedVenv "Scripts\python.exe"
$buildRoot = Join-Path $repoRoot "build\ClipSift"
$distFolder = Join-Path $repoRoot "dist\ClipSift"

if (-not $env:VIRTUAL_ENV) {
    throw "Activate ClipSift's .venv before building."
}
if ((Resolve-Path -LiteralPath $env:VIRTUAL_ENV).Path -ne (Resolve-Path -LiteralPath $expectedVenv).Path) {
    throw "The active virtual environment is not ClipSift's repository .venv."
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ClipSift's .venv Python was not found: $python"
}

$version = & $python -c "from clipsift import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0 -or -not $version) { throw "Could not determine the ClipSift version." }
$versionParts = @($version.Split('.') | ForEach-Object { [int]$_ })
while ($versionParts.Count -lt 4) { $versionParts += 0 }
$versionComma = ($versionParts[0..3] -join ', ')

$allowedCleanup = @(
    $buildRoot,
    $distFolder,
    (Join-Path $repoRoot "dist\ClipSift-$version-win64.zip"),
    (Join-Path $repoRoot "dist\ClipSift-$version-win64.zip.sha256"),
    (Join-Path $repoRoot "dist\packaged-smoke-check.json")
)
foreach ($target in $allowedCleanup) {
    $resolvedParent = [IO.Path]::GetFullPath((Split-Path -Parent $target))
    if (-not $resolvedParent.StartsWith([IO.Path]::GetFullPath($repoRoot), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup outside the repository: $target"
    }
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
}
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $repoRoot "dist") -Force | Out-Null

$template = Get-Content -Raw -LiteralPath (Join-Path $repoRoot "packaging\windows\version_info.txt.in")
$template.Replace("@VERSION_COMMA@", $versionComma).Replace("@VERSION@", $version) |
    Set-Content -LiteralPath (Join-Path $buildRoot "version_info.txt") -Encoding UTF8

Write-Host "Running complete test suite..."
& $python -m pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed; package was not built." }

Write-Host "Building ClipSift $version..."
& $python -m PyInstaller --noconfirm --distpath (Join-Path $repoRoot "dist") --workpath $buildRoot (Join-Path $repoRoot "ClipSift.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$executable = Join-Path $distFolder "ClipSift.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Expected executable was not created: $executable" }

$smokeOutput = Join-Path $repoRoot "dist\packaged-smoke-check.json"
$process = Start-Process -FilePath $executable -ArgumentList @("--packaged-smoke-check", $smokeOutput) -PassThru -WindowStyle Hidden
if (-not $process.WaitForExit(180000)) {
    Stop-Process -Id $process.Id -Force
    throw "Packaged offline smoke check exceeded three minutes."
}
if (-not (Test-Path -LiteralPath $smokeOutput -PathType Leaf)) {
    throw "Packaged offline smoke check failed."
}
$smoke = Get-Content -Raw -LiteralPath $smokeOutput | ConvertFrom-Json
$failure = if ($smoke.failure) { "$($smoke.failure.type): $($smoke.failure.message)" } else { "No structured failure detail was written." }
if ($process.ExitCode -ne 0 -or $smoke.success -ne $true) { throw "Packaged offline smoke check failed: $failure" }
$failedImports = @($smoke.imports.PSObject.Properties | Where-Object Value -ne "ok")
if ($failedImports.Count -gt 0 -or $smoke.model_loaded -ne $false) { throw "Packaged module verification failed." }

$folderBytes = (Get-ChildItem -LiteralPath $distFolder -Recurse -File | Measure-Object Length -Sum).Sum
$zipPath = Join-Path $repoRoot "dist\ClipSift-$version-win64.zip"
Compress-Archive -LiteralPath $distFolder -DestinationPath $zipPath -CompressionLevel Optimal
$hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
$checksumPath = "$zipPath.sha256"
"$($hash.Hash.ToLowerInvariant())  $([IO.Path]::GetFileName($zipPath))" | Set-Content -LiteralPath $checksumPath -Encoding ascii
$zipBytes = (Get-Item -LiteralPath $zipPath).Length

Write-Host "Build complete: $executable"
Write-Host ("Onedir size: {0:N2} GiB" -f ($folderBytes / 1GB))
Write-Host ("ZIP size: {0:N2} GiB" -f ($zipBytes / 1GB))
Write-Host "Checksum: $checksumPath"
