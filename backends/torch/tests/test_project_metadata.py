import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ProjectMetadataTests(unittest.TestCase):
    def test_project_name_is_rayd_torch(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(data["project"]["name"], "rayd-torch")

    def test_default_dependencies_require_torch_not_dr_jit(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        deps = [dep.lower() for dep in data["project"].get("dependencies", [])]
        self.assertTrue(any(dep.startswith("torch") for dep in deps))
        self.assertFalse(any(dep.startswith("dr" + "jit") for dep in deps))

    def test_transitional_wheels_cover_supported_python_and_torch_baseline(self):
        data = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(data["project"]["requires-python"], ">=3.10,<3.15")
        self.assertIn("torch>=2.10,<2.12", data["project"]["dependencies"])
        self.assertIn("torch==2.10.0", data["build-system"]["requires"])

    def test_public_python_source_has_no_obsolete_product_name(self):
        source_root = Path("python") / "rayd" / "torch"
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
        self.assertNotIn("ray" + "dn", source.lower())
        self.assertNotIn("rayd-native", source.lower())
        self.assertNotIn("_ray" + "dn", source.lower())

    def test_stable_abi_slice_avoids_unstable_torch_and_python_apis(self):
        stable_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("src/stable").glob("*.cu"))
        )
        cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
        for forbidden in ("at::", "c10::", "py::", "torch/extension.h", "torch/library.h"):
            self.assertNotIn(forbidden, stable_source)
        self.assertIn("STABLE_TORCH_LIBRARY(rayd_torch_stable", stable_source)
        self.assertIn("TORCH_TARGET_VERSION=0x020a000000000000", cmake)
        stable_start = cmake.index("rayd_torch_stable_ops\n        SHARED")
        stable_target = cmake[stable_start:cmake.index("execute_process(", stable_start)]
        self.assertNotIn("TORCH_PYTHON_LIBRARY", stable_target)
        self.assertNotIn('"${TORCH_LIBRARIES}"', stable_target)
        self.assertNotIn("CUDA::cuda_driver", stable_target)
        self.assertIn('"${RAYD_TORCH_STABLE_CPU_LIBRARY}"', stable_target)
        self.assertIn('"${RAYD_TORCH_STABLE_CUDA_LIBRARY}"', stable_target)
        self.assertIn("CUDA::cudart", stable_target)

    def test_stable_abi_audit_script_is_packaged_with_the_backend(self):
        script = Path("scripts/verify_stable_abi.py")
        self.assertTrue(script.is_file())
        source = script.read_text(encoding="utf-8")
        for dependency in ("torch_python", "c10.dll", "libc10.so", "python3"):
            self.assertIn(dependency, source)
        for symbol in ('"at::"', '"c10::"', '"@at@@"', '"@c10@@"'):
            self.assertIn(symbol, source)

    def test_local_cuda_build_targets_native_gpu(self):
        cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(pyproject["tool"]["scikit-build"]["build-dir"], "build/{wheel_tag}")
        self.assertIn('set(RAYD_TORCH_DEFAULT_CUDA_ARCHITECTURES "native")', cmake)
        self.assertIn("torch.cuda.get_device_capability()", cmake)
        self.assertIn("print(f'{major}.{minor}')", cmake)
        self.assertNotIn("print(f'{major}.{minor}+PTX')", cmake)
        self.assertIn("if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES}", cmake)
        self.assertIn("ENV{TORCH_CUDA_ARCH_LIST}", cmake)

        dev_build = Path("scripts/dev_build_native.ps1").read_text(encoding="utf-8")
        self.assertIn("envs\\witwin3\\python.exe", dev_build)
        for target in ("rayd_torch_stable_ops", "rayd_torch_legacy_ops", "_C"):
            self.assertIn(target, dev_build)
        for artifact in ("_stable_ops*.dll", "_legacy_ops*.dll", "_C*.pyd"):
            self.assertIn(artifact, dev_build)

        local_build = (Path(__file__).resolve().parents[3] / "scripts" / "build_local.ps1").read_text(
            encoding="utf-8"
        )
        for marker in (
            "CMAKE_BUILD_PARALLEL_LEVEL",
            'CMAKE_GENERATOR = "Ninja"',
            "RAYD_CUDA_GENCODE_ARCHES",
            "CMAKE_CUDA_ARCHITECTURES",
            "TORCH_CUDA_ARCH_LIST",
            'build/local-$CudaArch',
            "VsDevCmd.bat",
            "NVCC_CCBIN",
            "Get-Command cl.exe -ErrorAction Stop",
        ):
            self.assertIn(marker, local_build)
        self.assertTrue((Path(__file__).resolve().parents[3] / "scripts" / "build_local.cmd").is_file())

    def test_ci_cuda_fat_binary_covers_witwin_platform_matrix(self):
        root = Path(__file__).resolve().parents[3]
        expected_cmake = "70-real;75-real;80-real;86-real;87-real;89-real;90-real;100-real;101-real;120-real;120-virtual"
        expected_torch = "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
        pypi = (root / ".github/workflows/pypi.yml").read_text(encoding="utf-8")
        pull_request = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        stable = (root / ".github/workflows/stable-abi-ci.yml").read_text(encoding="utf-8")
        self.assertIn(expected_cmake, pypi)
        self.assertIn(expected_torch, pypi)
        self.assertIn("87-real;120-real;120-virtual", stable)
        self.assertIn("8.7;12.0+PTX", stable)
        self.assertIn("--expected-sass 87,120", stable)
        torch_linux_env = pypi.split("CIBW_ENVIRONMENT_LINUX:", 2)[2].split(
            "CIBW_REPAIR_WHEEL_COMMAND_LINUX:", 1
        )[0]
        self.assertIn(f'CMAKE_CUDA_ARCHITECTURES="{expected_cmake}"', torch_linux_env)
        self.assertIn(f'TORCH_CUDA_ARCH_LIST="{expected_torch}"', torch_linux_env)
        for python, tag in (
            ("3.10", "cp310"),
            ("3.11", "cp311"),
            ("3.12", "cp312"),
            ("3.13", "cp313"),
            ("3.14", "cp314"),
        ):
            self.assertIn(
                f'{{python-version: "{python}", cibw-build: "{tag}-manylinux_x86_64"}}',
                pypi,
            )
        self.assertIn(
            "name: release-rayd-torch-linux-py${{ matrix.python-version }}",
            pypi,
        )
        self.assertIn("name: release-rayd-torch-linux-py3.10", pypi)
        torch_verifier = "--stem _legacy_ops --stem _stable_ops"
        self.assertEqual(pypi.count(torch_verifier), 2)
        self.assertNotIn("--stem _C --stem _stable_ops", pypi)
        self.assertIn('CMAKE_BUILD_PARALLEL_LEVEL=4', pypi)
        self.assertIn('CMAKE_BUILD_PARALLEL_LEVEL=2', pypi)
        self.assertIn('CMAKE_CUDA_FLAGS=--threads=2', pypi)
        self.assertIn('CMAKE_CUDA_COMPILER_LAUNCHER=', pypi)
        self.assertIn('mozilla-actions/sccache-action@v0.0.10', pypi)
        for grouped_flag in (
            "--generate-code=arch=compute_70,code=[sm_70,sm_75]",
            "--generate-code=arch=compute_80,code=[sm_80,sm_86,sm_87,sm_89]",
            "--generate-code=arch=compute_90,code=sm_90",
            "--generate-code=arch=compute_100,code=[sm_100,sm_101]",
            "--generate-code=arch=compute_120,code=[sm_120,compute_120]",
        ):
            self.assertIn(grouped_flag, pypi)
        self.assertIn("Windows Torch full wheel build exceeded the 60-minute release limit.", pypi)
        cmake = (root / "backends/torch/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("RAYD_TORCH_CUDA_GENCODE_FLAGS", cmake)
        self.assertIn("rayd_torch_apply_cuda_gencode(rayd_torch_stable_ops)", cmake)
        self.assertIn("rayd_torch_apply_cuda_gencode(rayd_torch_native_core)", cmake)
        self.assertIn('CMAKE_CUDA_ARCHITECTURES: "87-real;120-real;120-virtual"', pull_request)
        self.assertIn("--expected-sass 87,120", pull_request)
        self.assertNotIn("self-hosted", pull_request)

    def test_explicit_torch_architecture_precedes_environment_and_gpu_detection(self):
        cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")
        explicit = cmake.index("if(DEFINED TORCH_CUDA_ARCH_LIST")
        environment = cmake.index("elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST}")
        detection = cmake.index("torch.cuda.get_device_capability()")
        self.assertLess(explicit, environment)
        self.assertLess(environment, detection)

    def test_multipath_pipeline_uses_current_optix_link_options(self):
        source = Path("src/torch_ext/common/optix_pipeline.cpp").read_text(encoding="utf-8")
        self.assertIn("link_options.maxTraceDepth = 1", source)
        self.assertIn("optixPipelineSetStackSize", source)
        for removed_field in (
            "maxContinuationCallableDepth",
            "maxDirectCallableDepthFromState",
            "maxDirectCallableDepthFromTraversal",
            "link_options.maxTraversableGraphDepth",
        ):
            self.assertNotIn(removed_field, source)

    def test_cuda_multipath_params_are_stream_local(self):
        source = Path("src/torch_ext/scene/multipath_cuda.cu").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("__constant__", source)
        self.assertNotIn("cudaMemcpyToSymbol", source)
        self.assertIn("extern __shared__", source)
        self.assertIn("getCurrentCUDAStream", source)

    def test_optix_auto_fallback_preserves_operational_errors(self):
        source = Path("src/torch_ext/scene/optix_context.cpp").read_text(
            encoding="utf-8"
        )
        for capability_error in (
            "OPTIX_ERROR_LIBRARY_NOT_FOUND",
            "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION",
            "OPTIX_ERROR_NOT_SUPPORTED",
            "OPTIX_ERROR_NOT_COMPATIBLE",
        ):
            self.assertIn(capability_error, source)
        for operational_error in (
            "OPTIX_ERROR_DEVICE_OUT_OF_MEMORY",
            "OPTIX_ERROR_CUDA_ERROR",
        ):
            self.assertNotIn(
                f"context_result == {operational_error}",
                source,
            )
