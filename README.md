# RayD

RayD is a CUDA/OptiX project with explicit Dr.Jit and Torch backends.

```python
import rayd.drjit as rd
import rayd.torch as rt
```

Install both backends through the file-free meta-distribution:

```bash
pip install rayd
```

Install only one backend when the other runtime is not needed:

```bash
pip install rayd-drjit
pip install rayd-torch
```

All three distributions share version `0.6.0`. The meta-distribution pins both
backend distributions to that same version. The parent `rayd` package remains
a PEP 420 namespace and does not select or re-export a default backend.

Backend sources, builds, and tests live in `backends/drjit` and
`backends/torch`. Cross-backend contracts, packaging checks, and coexistence
tests live at the repository root.

## Capabilities

| Capability | Dr.Jit | Torch |
| --- | --- | --- |
| Intersection, point/ray nearest edge | Yes | Yes |
| Top-K nearest edges | Yes | No |
| Visibility and reflection tracing | Yes | Yes |
| Reflection and diffraction accumulation | Yes | Yes |
| Reverse and forward AD | Yes | Yes |
| Surfel primitives | Yes | No |
| `torch.compile` integration | No | Yes |

Use `backend_capabilities()` on either subpackage for the machine-readable
manifest. Unsupported features do not silently cross runtimes.

## Runtime boundary

Each backend owns its Scene and Mesh objects, CUDA allocations, current-stream
behavior, OptiX context/pipelines/SBT/acceleration structures, and AD graph.
Objects and handles never cross the backend boundary. Shared code is limited to
semantic contracts, test data, and runtime-independent device headers.

## Breaking namespace release

This is a hard cut: the parent namespace exports no default backend, and the
former Torch package and dispatcher names are not registered or forwarded.
Downstream code must choose `rayd.drjit` or `rayd.torch` explicitly.
