# Examples

- [`basics/ray_mesh_intersection.py`](basics/ray_mesh_intersection.py)
  Custom batched rays against a simple mesh and prints hit / miss results.
- [`basics/nearest_edge_query.py`](basics/nearest_edge_query.py)
  Queries the closest mesh edge for a batch of custom points.
- [`basics/surfel_intersection.py`](basics/surfel_intersection.py)
  Traces a ray against an independent 2DGS-style surfel scene and differentiates hit distance with respect to surfel center.
- [`basics/surfel_multiview_color_fit.py`](basics/surfel_multiview_color_fit.py)
  Fits per-surfel RGB or degree-1 SH-like color from posed multi-view images using RayD surfel alpha compositing.
- [`renderer/cornell_box.py`](renderer/cornell_box.py)
  Renders a compact colored Cornell Box with simple direct lighting, using an example-local camera helper.
