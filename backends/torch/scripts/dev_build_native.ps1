param(
    [string]$PythonExe = "C:\Users\Asixa\miniconda3\envs\witwin2\python.exe",
    [string]$BuildDir = "artifacts/skbuild",
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
& $CMakeExe --build $BuildPath --config $Config --target _C
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$BuiltExtension = Get-ChildItem (Join-Path $BuildPath $Config) -Filter "_C*.pyd" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $BuiltExtension) {
    throw "Built extension not found under $(Join-Path $BuildPath $Config)"
}

$Destination = & $PythonExe -c "import pathlib, sysconfig; root = pathlib.Path(sysconfig.get_path('platlib')) / 'rayd' / 'torch'; matches = sorted(root.glob('_C*.pyd')); print(matches[0] if matches else '')"
if (-not $Destination) {
    throw "Could not resolve currently installed rayd.torch._C destination."
}

Copy-Item -LiteralPath $BuiltExtension.FullName -Destination $Destination -Force
$Stopwatch.Stop()

Write-Host "Built: $($BuiltExtension.FullName)"
Write-Host "Copied: $Destination"
Write-Host ("Elapsed seconds: {0:N2}" -f $Stopwatch.Elapsed.TotalSeconds)
