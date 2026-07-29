# Copyright Xingyu Chen.
# Demonstrates surfel intersection.

import rayd.drjit as rd
import drjit as dr
import drjit.cuda.ad as ad


def make_scene(center_z: float = 0.0) -> rd.SurfelScene:
    opts = rd.SurfelTraceOptions()
    opts.primitive_mode = rd.SurfelPrimitiveMode.Icosahedron20

    cloud = rd.SurfelCloud(
        dr.cuda.Array3f([0.0], [0.0], [center_z]),
        dr.cuda.Array3f([1.0], [0.0], [0.0]),
        dr.cuda.Array3f([0.0], [1.0], [0.0]),
        dr.cuda.Float([1.0]),
    )
    scene = rd.SurfelScene(cloud, opts)
    scene.build()
    return scene


def main() -> None:
    scene = make_scene()
    ray = rd.Ray(dr.cuda.Array3f([0.25], [0.25], [1.0]), dr.cuda.Array3f([0.0], [0.0], [-1.0]))
    its = scene.intersect(ray)
    print("Surfel ray intersection example")
    print(
        f"hit={bool(its.is_valid()[0])}, "
        f"t={float(its.t[0]):.4f}, "
        f"surfel_id={int(its.surfel_id[0])}, "
        f"local_uv=({float(its.local_uv[0][0]):.4f}, {float(its.local_uv[1][0]):.4f}), "
        f"gaussian_weight={float(its.gaussian_weight[0]):.4f}"
    )

    center_z = ad.Float([0.0])
    dr.enable_grad(center_z)
    ad_cloud = rd.SurfelCloud(
        ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), center_z),
        ad.Array3f(ad.Float([1.0]), ad.Float([0.0]), ad.Float([0.0])),
        ad.Array3f(ad.Float([0.0]), ad.Float([1.0]), ad.Float([0.0])),
        ad.Float([1.0]),
    )
    ad_scene = rd.SurfelScene(ad_cloud)
    ad_scene.build()
    ad_ray = rd.RayAD(
        ad.Array3f(ad.Float([0.25]), ad.Float([0.25]), ad.Float([1.0])),
        ad.Array3f(ad.Float([0.0]), ad.Float([0.0]), ad.Float([-1.0])),
    )
    ad_its = ad_scene.intersect(ad_ray)
    dr.backward(dr.sum(ad_its.t))
    print(f"d hit_t / d center_z = {float(dr.grad(center_z)[0]):.4f}")


if __name__ == "__main__":
    main()
