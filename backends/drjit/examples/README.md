# Examples

- [`basics/ray_mesh_intersection.py`](basics/ray_mesh_intersection.py)
  Custom batched rays against a simple mesh and prints hit / miss results.
- [`basics/nearest_edge_query.py`](basics/nearest_edge_query.py)
  Queries the closest mesh edge for a batch of custom points.
- [`basics/surfel_intersection.py`](basics/surfel_intersection.py)
  Traces a ray against an independent 2DGS-style surfel scene and differentiates hit distance with respect to surfel center.
- [`basics/surfel_multiview_color_fit.py`](basics/surfel_multiview_color_fit.py)
  Fits per-surfel RGB or degree-1 SH-like color from posed multi-view images using RayD surfel alpha compositing.
- [`basics/surfel_convergence_video.py`](basics/surfel_convergence_video.py)
  Renders a surfel-fit convergence animation. Flags: `--size`, `--iterations`, `--frame-step`, `--output-dir`.
- [`basics/surfel_fit_web_image.py`](basics/surfel_fit_web_image.py)
  Fits a surfel cloud to an image. Flags: `--input` (required), `--output-dir`, `--size`, `--source-url`, `--no-autocontrast`.
- [`renderer/cornell_box.py`](renderer/cornell_box.py)
  Renders a colored Cornell Box with a multi-bounce Monte Carlo path tracer and a forward-mode AD gradient heatmap, using an example-local camera helper.
