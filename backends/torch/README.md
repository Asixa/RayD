# RayDTorch

RayDTorch is a Torch-native CUDA/OptiX package for RayD geometry and multipath primitives.

```python
import raydtorch as rt
```

The public package name is `raydtorch`; it does not provide `rayd` compatibility aliases, so it can coexist with the original RayD package in the same environment.

## Tensor ABI

RayDTorch APIs accept CUDA `torch.float32` tensors for vector data and CUDA `torch.int32` tensors for index data. Vector tensors are row-major `(N, 3)` unless otherwise documented, masks are `torch.bool`, and tensors should be contiguous. Outputs and AD tapes are Torch-owned tensors.

## Gradient Contract

Intersection, edge, reflection, EPC, and diffraction operators use a fixed-winner gradient contract. The discrete primitive, edge, visibility, or path decision selected in the forward pass is treated as non-differentiable; VJP and JVP propagate through the continuous values recomputed from the saved winner and live Torch tensors.

## Autograd

The native operators support Torch reverse-mode VJP and forward-mode JVP for the supported continuous inputs. CUDA work is launched on the current Torch CUDA stream.

## Dependencies

RayDTorch depends on PyTorch, CUDA, and OptiX for native execution. The RayDTorch package path has no Dr.Jit dependency.
