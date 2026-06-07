# RayDTorch Agent Rules

## Native CUDA/OptiX Incremental Build

- Do not use `python -m pip install --no-build-isolation -e .` for every debug iteration. That command creates or refreshes a full editable build/install flow and is too slow for CUDA/OptiX work.
- The persistent native build directory is `artifacts/skbuild`. It is ignored by git and should be reused across edits.
- If `artifacts/skbuild` does not exist, initialize it once:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m pip install --no-build-isolation -e . -Cbuild-dir=artifacts/skbuild
```

- After native `.cpp`, `.cu`, `.h`, CMake, or PTX embedding changes, use the incremental helper:

```powershell
powershell -ExecutionPolicy Bypass -File E:\Code\RayDTorch\scripts\dev_build_native.ps1
```

- The helper runs `cmake --build artifacts/skbuild --config Release --target _raydtorch` and copies the resulting `_raydtorch*.pyd` to the conda site-packages path that the editable import hook actually loads.
- Run focused tests with the environment Python directly, for example:

```powershell
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest tests.raydtorch_native.test_multipath -v
C:\Users\Asixa\miniconda3\envs\witwin2\python.exe -m unittest discover tests.raydtorch_native -v
```

- Use full `pip install -e .` again only when intentionally regenerating the editable install metadata, changing packaging behavior, or recreating the persistent build directory from scratch.
