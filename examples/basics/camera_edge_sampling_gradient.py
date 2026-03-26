from pathlib import Path

import rayd as rd
import drjit as dr


def make_cube_scene(tx: dr.cuda.ad.Float) -> rd.Scene:
    mesh = rd.Mesh(
        dr.cuda.Array3f(
            [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
            [3.5, 3.5, 3.5, 3.5, 4.5, 4.5, 4.5, 4.5],
        ),
        dr.cuda.Array3i(
            [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3],
            [2, 3, 5, 6, 7, 4, 2, 6, 1, 5, 7, 6],
            [1, 2, 6, 7, 3, 7, 6, 5, 5, 4, 6, 2],
        ),
    )
    mesh.to_world_left = dr.cuda.ad.Matrix4f(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    scene = rd.Scene()
    scene.add_mesh(mesh)
    scene.build()
    return scene


def make_camera(width: int, height: int) -> rd.Camera:
    camera = rd.Camera(45.0, 1e-4, 1e4)
    camera.width = width
    camera.height = height
    camera.build()
    return camera


def save_gradient_image(path: Path, image: dr.cuda.ad.TensorXf) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "This example saves a PNG and requires matplotlib and numpy."
        ) from exc

    values = np.asarray(list(dr.detach(image.array)), dtype=np.float32).reshape(image.shape)
    limit = float(np.max(np.abs(values))) if values.size else 1.0
    limit = limit if limit > 0.0 else 1.0

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("Primary-Edge Gradient d(depth)/d(tx)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    width = 128
    height = 128

    tx = dr.cuda.ad.Float([0.0])
    dr.enable_grad(tx)

    scene = make_cube_scene(tx)
    camera = make_camera(width, height)

    edge_image = camera.render_grad(scene, spp=8)

    dr.set_grad(tx, 1.0)
    dr.forward_to(edge_image)
    grad_image = dr.grad(edge_image)

    output_path = Path(__file__).resolve().parents[1] / "build" / "examples" / "camera_edge_sampling_gradient.png"
    save_gradient_image(output_path, grad_image)

    grad_values = [float(v) for v in list(dr.detach(grad_image.array))]
    print("Camera edge-sampling gradient example")
    print("type =", type(grad_image).__name__)
    print("shape =", grad_image.shape)
    print("abs_max =", max(abs(v) for v in grad_values))
    print("nonzero =", sum(1 for v in grad_values if abs(v) > 1e-8))
    print("saved =", output_path)


if __name__ == "__main__":
    main()
