#!/usr/bin/env bash
# Copyright Xingyu Chen.
# Builds and editable-installs one or both RayD backends from this checkout on Linux.

set -euo pipefail

usage() {
    cat <<'EOF'
usage: scripts/build_local.sh [drjit|torch|all] [--cuda-arch ARCHS] [--python PYTHON] [--fetch-optix-headers] [--clean]

  drjit|torch|all        backend to build (default: all)
  --cuda-arch ARCHS      RAYD_CUDA_ARCHS value: native (default), 86, 75,86,89, or 8.9+PTX
  --python PYTHON        interpreter of the target environment (default: python)
  --fetch-optix-headers  let the Torch build download the OptiX 8.1 headers when none are found
  --clean                delete this build's CMake directory first

The environment must already hold the backend's framework: drjit==1.3.1 for rayd-drjit, or a CUDA build of
torch 2.10 for rayd-torch (python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128),
plus scikit-build-core, ninja, and cmake, and a CUDA 12.x toolkit with nvcc (a conda cuda-toolkit works).
EOF
}

backend=all
cuda_arch=${RAYD_CUDA_ARCHS:-native}
python=python
fetch_optix=0
clean=0
while [ $# -gt 0 ]; do
    case "$1" in
        drjit | torch | all) backend=$1 ;;
        --cuda-arch) cuda_arch=${2:?--cuda-arch needs a value}; shift ;;
        --python) python=${2:?--python needs a value}; shift ;;
        --fetch-optix-headers) fetch_optix=1 ;;
        --clean) clean=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "build_local.sh: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if ! [[ $cuda_arch =~ ^[0-9A-Za-z.,\;+\ ]+$ ]]; then
    echo "Invalid CUDA architecture list '$cuda_arch'. Use native or values such as 86, 75,86, or 8.9+PTX." >&2
    exit 2
fi

if [ -z "${CUDACXX:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc was not found. Install a CUDA 12.x toolkit and put its bin directory on PATH (or set CUDACXX)." >&2
    exit 1
fi

# The release-only architecture variables would conflict with RAYD_CUDA_ARCHS, which drives every CUDA target of a
# source build in both backends; CMake refuses the combination.
unset RAYD_CUDA_GENCODE_ARCHES RAYD_CUDA_PTX_ARCH CMAKE_CUDA_ARCHITECTURES TORCH_CUDA_ARCH_LIST
unset RAYD_TORCH_CUDA_GENCODE_FLAGS
export RAYD_CUDA_ARCHS=$cuda_arch

if command -v ninja >/dev/null 2>&1 || "$python" -c "import ninja" >/dev/null 2>&1; then
    export CMAKE_GENERATOR=${CMAKE_GENERATOR:-Ninja}
fi
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}
build_tag=$(printf '%s' "$cuda_arch" | tr '[:upper:]' '[:lower:]' | tr -cs '0-9a-z' '-' | sed 's/^-*//; s/-*$//')

optix_headers_found() {
    local candidate
    for candidate in "${OPTIX_INCLUDE_DIR:-}" "${OPTIX_PATH:+$OPTIX_PATH/include}" \
        /usr/local/NVIDIA-OptiX-SDK-*/include /opt/NVIDIA-OptiX-SDK-*/include; do
        if [ -n "$candidate" ] && [ -f "$candidate/optix.h" ]; then
            return 0
        fi
    done
    return 1
}

build_backend() {
    local name=$1
    local -a extra=()
    if [ "$name" = torch ]; then
        if ! "$python" -c "import sys, torch; sys.exit(torch.version.cuda is None)" >/dev/null 2>&1; then
            echo "rayd-torch needs a CUDA build of torch in this environment ($python). Install it with:" >&2
            echo "    $python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128" >&2
            exit 1
        fi
        if ! optix_headers_found; then
            if [ "$fetch_optix" = 1 ]; then
                extra+=("-Ccmake.define.RAYD_FETCH_OPTIX_HEADERS=ON")
            else
                cat >&2 <<'EOF'
rayd-torch compiles its OptiX programs and needs the OptiX SDK 8.1 (or newer) headers, which were not found.
Set OPTIX_PATH (or OPTIX_INCLUDE_DIR) to an SDK, for example the headers release CI uses:
    git clone --depth 1 --branch v8.1.0 https://github.com/NVIDIA/optix-dev.git ~/optix-dev
    export OPTIX_PATH=~/optix-dev
or rerun this script with --fetch-optix-headers to let CMake download them. rayd-drjit needs no OptiX SDK.
EOF
                exit 1
            fi
        fi
    fi
    local build_dir="build/local-$build_tag"
    if [ "$clean" = 1 ] && [ -d "$repo_root/$name/$build_dir" ]; then
        rm -rf -- "${repo_root:?}/$name/$build_dir"
    fi
    echo "Building RayD $name for RAYD_CUDA_ARCHS=$cuda_arch with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs"
    # --no-build-isolation builds against the drjit or torch already installed here. An isolated build would download
    # its own pinned copy instead, which for Torch is a CPU-only wheel from PyPI and cannot build the extension.
    "$python" -m pip install --no-user --no-build-isolation -ve "$repo_root/$name" "-Cbuild-dir=$build_dir" \
        ${extra[@]+"${extra[@]}"}
}

case "$backend" in
    drjit) build_backend drjit ;;
    torch) build_backend torch ;;
    all) build_backend drjit; build_backend torch ;;
esac
