# RayD

RayD is a CUDA/OptiX project with explicit Dr.Jit and Torch backends.

```python
import rayd.drjit as rd
import rayd.torch as rt
```

Install either backend independently with `rayd-drjit` or `rayd-torch`.
The parent `rayd` package is a PEP 420 namespace and does not select a default
backend.

Backend sources, builds, and tests live in `backends/drjit` and
`backends/torch`. Cross-backend contracts, packaging checks, and coexistence
tests live at the repository root.
