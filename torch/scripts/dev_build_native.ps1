param(
    [string]$PythonExe = "C:\Users\Asixa\miniconda3\envs\witwin3\python.exe",
    [string]$BuildDir = "build/local-120",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildPath = Join-Path $RepoRoot $BuildDir
if (-not (Test-Path $BuildPath)) {
    throw "Build directory not found: $BuildPath. Run: $PythonExe -m pip install --no-build-isolation -e . -Cbuild-dir=$BuildDir"
}

$EnvRoot = Split-Path -Parent $PythonExe
$CMakeExe = Join-Path $EnvRoot "Scripts\cmake.exe"
if (-not (Test-Path $CMakeExe)) {
    $CMakeExe = Join-Path $EnvRoot "Lib\site-packages\cmake\data\bin\cmake.exe"
}
if (-not (Test-Path $CMakeExe)) {
    throw "cmake.exe not found next to Python environment: $PythonExe"
}

$Stopwatch = [Diagnostics.Stopwatch]::StartNew()
& $CMakeExe --build $BuildPath --config $Config --target rayd_torch_stable_ops rayd_torch_legacy_ops _C
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$InstallRoot = & $PythonExe -c "import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_path('platlib')) / 'rayd' / 'torch')"
if (-not $InstallRoot -or -not (Test-Path $InstallRoot)) {
    throw "Could not resolve the installed rayd.torch package directory."
}

$Artifacts = @(
    @{ Pattern = "_stable_ops*.dll"; Label = "Stable ABI library" },
    @{ Pattern = "_legacy_ops*.dll"; Label = "legacy operator library" },
    @{ Pattern = "_C*.pyd"; Label = "metadata compatibility shim" }
)
foreach ($Artifact in $Artifacts) {
    $BuiltArtifact = Get-ChildItem (Join-Path $BuildPath $Config) -Filter $Artifact.Pattern |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $BuiltArtifact) {
        throw "$($Artifact.Label) not found under $(Join-Path $BuildPath $Config)"
    }
    $Destination = Join-Path $InstallRoot $BuiltArtifact.Name
    Copy-Item -LiteralPath $BuiltArtifact.FullName -Destination $Destination -Force
    Write-Host "Copied: $Destination"
}
$Stopwatch.Stop()

Write-Host ("Elapsed seconds: {0:N2}" -f $Stopwatch.Elapsed.TotalSeconds)
