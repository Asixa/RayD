import rayd as rd
import drjit.cuda as cuda


def make_scene() -> tuple[rd.Scene, rd.Mesh]:
    mesh = rd.Mesh(
        cuda.Array3f(
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ),
        cuda.Array3i([0, 0], [1, 2], [2, 3]),
    )
    mesh.configure()

    scene = rd.Scene()
    scene.add_mesh(mesh)
    scene.configure()
    return scene, mesh


def main() -> None:
    scene, mesh = make_scene()

    secondary_edges = mesh.secondary_edges()
    boundary_count = sum(bool(v) for v in list(secondary_edges.is_boundary))

    queries = cuda.Array3f(
        [0.25, 1.20, -0.15, 0.50],
        [0.25, 0.50, 0.70, 1.20],
        [0.10, 0.00, 0.00, 0.00],
    )
    result = scene.nearest_edge(queries)

    print("Nearest-edge query example")
    print("secondary_edge_count =", secondary_edges.size())
    print("boundary_edge_count =", boundary_count)

    for i, valid in enumerate(list(result.is_valid())):
        if valid:
            edge_point = [float(result.edge_point[axis][i]) for axis in range(3)]
            print(
                f"query[{i}] valid: distance={float(result.distance[i]):.4f}, "
                f"shape_id={int(result.shape_id[i])}, edge_id={int(result.edge_id[i])}, "
                f"boundary={bool(result.is_boundary[i])}, edge_point={edge_point}"
            )
        else:
            print(f"query[{i}] invalid")


if __name__ == "__main__":
    main()

