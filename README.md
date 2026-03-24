# RayD

RayD is a minimalist differentiable ray tracing package built on top of Dr.Jit and OptiX.

```bash
pip install rayd
```

RayD is not a full renderer. It is a thin wrapper around Dr.Jit and OptiX for building your own renderers and simulators.

The goal is simple: expose differentiable ray-mesh intersection on the GPU without bringing in a full graphics framework.

## Why RayD?

RayD is for users who want OptiX acceleration and autodiff, but do not want a full renderer.

Why not Mitsuba? Mitsuba is excellent for graphics rendering, but often too high-level for RF, acoustics, sonar, or custom wave simulation. In those settings, direct access to ray-scene queries and geometry gradients is usually more useful than a full material-light-integrator stack.

RayD keeps only the geometric core:

- differentiable ray-mesh intersection
- scene-level GPU acceleration through OptiX
- edge acceleration structures for nearest-edge queries
- primary-edge sampling support for edge-based gradient terms

For intersection workloads, RayD targets Mitsuba-level performance and matching results with a much smaller API surface.

## What RayD Provides

- `Mesh`: triangle geometry, transforms, UVs, and edge topology
- `Scene`: a container of meshes plus OptiX acceleration
- `scene.intersect(ray)`: differentiable ray-mesh intersection
- `scene.nearest_edge(query)`: nearest-edge queries for points and rays
- edge acceleration data that is useful for edge sampling and edge diffraction methods

## Quick Examples

If you only want to see the package in action, start here:

- [`examples/basics/ray_mesh_intersection.py`](examples/basics/ray_mesh_intersection.py): custom rays against a mesh
- [`examples/basics/nearest_edge_query.py`](examples/basics/nearest_edge_query.py): nearest-edge queries
- [`examples/basics/camera_edge_sampling_gradient.py`](examples/basics/camera_edge_sampling_gradient.py): camera-driven edge-sampling gradients

Build meshes, put them in a scene, launch rays, define a loss, and backpropagate through geometry.

## Minimal Differentiable Ray Tracing Example

The example below traces a single ray against one triangle and backpropagates the hit distance to the vertex positions.

```python
import rayd as rd
import drjit as dr
import drjit.cuda as cuda
import drjit.cuda.ad as ad


mesh = rd.Mesh(
    cuda.Array3f([0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]),
    cuda.Array3i([0], [1], [2]),
)

verts = ad.Array3f(
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
)
dr.enable_grad(verts)

mesh.vertex_positions = verts

scene = rd.Scene()
scene.add_mesh(mesh)
scene.configure()

ray = rd.Ray(
    ad.Array3f([0.25], [0.25], [-1.0]),
    ad.Array3f([0.0], [0.0], [1.0]),
)

its = scene.intersect(ray)
loss = dr.sum(its.t)
dr.backward(loss)

print("t =", its.t)
print("grad z =", dr.grad(verts)[2])
```

This is the core RayD workflow. Replace the single ray with your own batched rays, RF paths, acoustic paths, or edge-based objectives.

## Edge Acceleration Structure

RayD also provides a scene-level edge acceleration structure.

This is useful for:

- edge sampling
- nearest-edge queries
- visibility-boundary terms
- geometric edge diffraction models

In other words, RayD is not limited to triangle hits. It also gives you direct access to edge-level geometry queries, which are important in many non-graphics simulators.

## Compiling Locally

RayD is a Python package with a C++/CUDA extension.

You need Python `>=3.10`, CUDA Toolkit `>=11.0`, CMake, a C++17 compiler, `drjit>=1.3.0`, `nanobind==2.11.0`, and `scikit-build-core`.

On Windows, use Visual Studio 2022 with Desktop C++ tools. On Linux, use GCC or Clang with C++17 support.

### Recommended environment

```powershell
conda create -n myenv python=3.10 -y
conda activate myenv
python -m pip install -U pip setuptools wheel
python -m pip install cmake scikit-build-core nanobind==2.11.0
python -m pip install "drjit>=1.3.0"
```

### Install

```powershell
conda activate myenv
python -m pip install .
```

## Dependencies

RayD depends on:

- Python `3.10+`
- Dr.Jit `1.3.0+`
- OptiX `8+`

RayD does not include:

- BSDFs
- emitters
- integrators
- scene loaders
- image I/O
- path tracing infrastructure

That is by design.

## Repository Layout

- [`include/rayd/`](include/rayd): public C++ headers
- [`src/`](src): C++ and CUDA implementation
- [`src/rayd.cpp`](src/rayd.cpp): Python bindings
- [`examples/`](examples): basic and renderer-side examples
- [`tests/test_geometry.py`](tests/test_geometry.py): geometry regression tests
- [`docs/api_reference.md`](docs/api_reference.md): Python API reference

## Testing

```powershell
python -m unittest tests.test_geometry -v
```

## Credits

RayD is developed with reference to:

- [psdr-jit](https://github.com/andyyankai/psdr-jit)
- [redner](https://github.com/BachiLi/redner)
- [mitsuba3](https://github.com/mitsuba-renderer/mitsuba3)

## Citation

```bibtex
@inproceedings{chen2026rfdt,
  title     = {Physically Accurate Differentiable Inverse Rendering
               for Radio Frequency Digital Twin},
  author    = {Chen, Xingyu and Zhang, Xinyu and Zheng, Kai and
               Fang, Xinmin and Li, Tzu-Mao and Lu, Chris Xiaoxuan
               and Li, Zhengxiong},
  booktitle = {Proceedings of the 32nd Annual International Conference
               on Mobile Computing and Networking (MobiCom)},
  year      = {2026},
  doi       = {10.1145/3795866.3796686},
  publisher = {ACM},
  address   = {Austin, TX, USA},
}
```

## License

MIT
