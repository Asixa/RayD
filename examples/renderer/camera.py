from dataclasses import dataclass
import math

import drjit as dr
import rayd as rd


EPSILON = 1e-4


@dataclass
class EdgeSample:
    x_dot_n: object
    idx: object
    ray_n: object
    ray_p: object
    pdf: object


class ExampleCamera:
    def __init__(self, fov_x=45.0, near_clip=1e-4, far_clip=1e4, width=512, height=512):
        self.fov_x = float(fov_x)
        self.near_clip = float(near_clip)
        self.far_clip = float(far_clip)
        self.width = int(width)
        self.height = int(height)
        self._edge_info = None
        self._edge_count = 0

    @property
    def _aspect(self):
        return float(self.width) / float(self.height)

    @property
    def _tan_half_fov(self):
        return math.tan(math.radians(self.fov_x) * 0.5)

    def prepare_edges(self, scene):
        self._edge_info = scene.edge_info()
        self._edge_count = self._edge_info.size()

    def sample_ray(self, samples):
        count = dr.width(samples[0])
        tan_half = self._tan_half_fov
        direction = dr.normalize(
            dr.cuda.Array3f(
                (1.0 - 2.0 * samples[0]) * tan_half,
                (1.0 - 2.0 * samples[1]) * tan_half / self._aspect,
                dr.full(dr.cuda.Float, 1.0, count),
            )
        )
        return rd.Ray(dr.zeros(dr.cuda.Array3f, count), direction)

    def sample_edge(self, sample1):
        count = dr.width(sample1)
        invalid = dr.full(dr.cuda.Int, -1, count)
        zero = dr.zeros(dr.cuda.Float, count)
        zero_ad = dr.zeros(dr.cuda.ad.Float, count)
        empty_ray = rd.Ray(dr.zeros(dr.cuda.Array3f, count), dr.zeros(dr.cuda.Array3f, count))

        if self._edge_info is None:
            raise RuntimeError("ExampleCamera.sample_edge(): call prepare_edges(scene) first.")
        if self._edge_count == 0:
            return EdgeSample(zero_ad, invalid, empty_ray, empty_ray, zero)

        scaled = sample1 * float(self._edge_count)
        edge_index_float = dr.minimum(dr.floor(scaled), float(self._edge_count - 1))
        edge_index = dr.cuda.UInt(edge_index_float)
        edge_t = scaled - edge_index_float

        p0 = dr.gather(dr.cuda.ad.Array3f, self._edge_info.start, edge_index)
        p1 = p0 + dr.gather(dr.cuda.ad.Array3f, self._edge_info.edge, edge_index)
        s0 = self._project(p0)
        s1 = self._project(p1)

        segment = s1 - s0
        segment_len = dr.maximum(dr.norm(dr.detach(segment)), EPSILON)
        segment_dir = dr.detach(segment) / segment_len
        edge_normal = dr.cuda.Array2f(-segment_dir[1], segment_dir[0])
        sample_pos_ad = s0 * (1.0 - edge_t) + s1 * edge_t
        sample_pos = dr.detach(sample_pos_ad)

        pixel_x = dr.cuda.Int(dr.floor(sample_pos[0] * float(self.width)))
        pixel_y = dr.cuda.Int(dr.floor(sample_pos[1] * float(self.height)))
        valid = (pixel_x >= 0) & (pixel_x < self.width) & (pixel_y >= 0) & (pixel_y < self.height)
        pixel_idx = pixel_y * self.width + pixel_x
        idx = dr.select(valid, pixel_idx, invalid)
        pdf = dr.select(valid, dr.rcp(float(self._edge_count) * segment_len), zero)
        x_dot_n = dr.select(valid, dr.dot(sample_pos_ad, edge_normal), zero_ad)

        ray_p = self.sample_ray(sample_pos + EPSILON * edge_normal)
        ray_n = self.sample_ray(sample_pos - EPSILON * edge_normal)
        return EdgeSample(x_dot_n, idx, ray_n, ray_p, pdf)

    def _project(self, point):
        safe_z = dr.maximum(point[2], EPSILON)
        tan_half = self._tan_half_fov
        return dr.cuda.ad.Array2f(
            0.5 - 0.5 * (point[0] / safe_z) / tan_half,
            0.5 - 0.5 * self._aspect * (point[1] / safe_z) / tan_half,
        )
