import rayd as rd
import drjit as dr


def make_scene() -> rd.Scene:
    mesh = rd.Mesh(
        dr.cuda.Array3f(
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 3.0],
        ),
        dr.cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )

    scene = rd.Scene()
    scene.add_mesh(mesh)
    scene.build()
    return scene


def main() -> None:
    scene = make_scene()

    rays = rd.RayDetached(
        dr.cuda.Array3f(
            [0.0, 0.75, 1.5, -0.5],
            [0.0, 0.75, 0.0, -0.5],
            [-1.0, -1.0, -1.0, -1.0],
        ),
        dr.cuda.Array3f(
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
    )

    its = scene.intersect(rays)

    print("Custom ray-mesh intersection example")
    for i, valid in enumerate(list(its.is_valid())):
        if valid:
            point = [float(its.p[axis][i]) for axis in range(3)]
            print(
                f"ray[{i}] hit: t={float(its.t[i]):.4f}, "
                f"shape_id={int(its.shape_id[i])}, prim_id={int(its.prim_id[i])}, p={point}"
            )
        else:
            print(f"ray[{i}] miss")


if __name__ == "__main__":
    main()
