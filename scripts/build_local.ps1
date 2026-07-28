param(
    [ValidateSet("drjit", "torch", "all")]
    [string]$Backend = "all",
    [string]$PythonExe = "python",
    [string]$CudaArch = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Initialize-MSVCEnvironment {
    if ($env:OS -ne "Windows_NT" -or (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        return
    }

    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        throw "vswhere.exe was not found. Install Visual Studio 2022 Desktop development with C++."
    }
    $InstallPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $VsDevCmd = Join-Path $InstallPath "Common7/Tools/VsDevCmd.bat"
    if (-not $InstallPath -or -not (Test-Path $VsDevCmd)) {
        throw "Visual Studio 2022 C++ build tools were not found."
    }

    $ImportedNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    cmd.exe /d /s /c "call `"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" |
        ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$" -and $ImportedNames.Add($Matches[1])) {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "VsDevCmd completed without placing cl.exe on PATH."
    }
}

Initialize-MSVCEnvironment

if ($env:OS -eq "Windows_NT") {
    $ClCommand = Get-Command cl.exe -ErrorAction Stop
    $env:NVCC_CCBIN = Split-Path -Parent $ClCommand.Source
}

if (-not $CudaArch) {
    $ComputeCapability = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
        Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or -not $ComputeCapability) {
        throw "Could not detect the local CUDA architecture. Pass -CudaArch explicitly (for example 120)."
    }
    $CudaArch = $ComputeCapability.Trim().Replace(".", "")
}
if ($CudaArch -notmatch "^[0-9]+$") {
    throw "Invalid CUDA architecture '$CudaArch'. Use digits such as 120."
}

$Parallelism = [Math]::Max(1, [Environment]::ProcessorCount)
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Parallelism"
$env:CMAKE_GENERATOR = "Ninja"

function Build-Backend([string]$Name) {
    $BackendRoot = Join-Path $RepoRoot "$Name"
    $BuildDir = "build/local-$CudaArch"
    $BuildPath = Join-Path $BackendRoot $BuildDir

    if ($Clean -and (Test-Path $BuildPath)) {
        $ResolvedBackend = (Resolve-Path $BackendRoot).Path
        $ResolvedBuild = (Resolve-Path $BuildPath).Path
        if (-not $ResolvedBuild.StartsWith($ResolvedBackend + [IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to remove build directory outside backend: $ResolvedBuild"
        }
        Remove-Item -LiteralPath $ResolvedBuild -Recurse -Force
    }

    if ($Name -eq "drjit") {
        $env:RAYD_CUDA_GENCODE_ARCHES = $CudaArch
        Remove-Item Env:RAYD_CUDA_PTX_ARCH -ErrorAction SilentlyContinue
    } else {
        $env:CMAKE_CUDA_ARCHITECTURES = $CudaArch
        $Major = [int]($CudaArch.Substring(0, $CudaArch.Length - 1))
        $Minor = $CudaArch.Substring($CudaArch.Length - 1)
        $env:TORCH_CUDA_ARCH_LIST = "$Major.$Minor"
    }

    Write-Host "Building RayD $Name for sm_$CudaArch with $Parallelism parallel jobs"
    & $PythonExe -m pip install --no-user --no-build-isolation -ve $BackendRoot "-Cbuild-dir=$BuildDir"
    if ($LASTEXITCODE -ne 0) {
        throw "RayD $Name build failed with exit code $LASTEXITCODE."
    }
}

if ($Backend -in @("drjit", "all")) {
    Build-Backend "drjit"
}
if ($Backend -in @("torch", "all")) {
    Build-Backend "torch"
}
