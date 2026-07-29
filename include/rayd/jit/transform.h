// Copyright Xingyu Chen.
// Declares the Dr.Jit transform API.

#pragma once

#include <rayd/jit/core.h>

#include <drjit/matrix.h>
#include <drjit/quaternion.h>
#include <drjit/sphere.h>
#include <drjit/transform.h>

namespace rayd {

/// Builders for 4x4 homogeneous transforms (column-vector convention, applied on the left).
namespace transform {

/// Translation matrix by \p vector.
template <typename Float_>
drjit::Matrix<Float_, 4> translate(const drjit::Array<Float_, 3> &vector) {
    return drjit::translate<drjit::Matrix<Float_, 4>>(vector);
}

/// Non-uniform scale matrix with per-axis factors \p vector.
template <typename Float_>
drjit::Matrix<Float_, 4> scale(const drjit::Array<Float_, 3> &vector) {
    return drjit::scale<drjit::Matrix<Float_, 4>>(vector);
}

/// Rotation matrix about \p axis by \p angle_degrees (degrees).
template <typename Float_>
drjit::Matrix<Float_, 4> rotate(const drjit::Array<Float_, 3> &axis, float angle_degrees) {
    return drjit::rotate<drjit::Matrix<Float_, 4>>(axis, drjit::deg_to_rad(angle_degrees));
}

/// Perspective projection from a vertical field of view and near/far clip planes.
inline ScalarMatrix4f perspective(float fov_degrees, float near_clip, float far_clip) {
    const float reciprocal_depth = 1.f / (far_clip - near_clip);
    const float tangent = drjit::tan(drjit::deg_to_rad(fov_degrees * 0.5f));
    const float cotangent = 1.f / tangent;

    ScalarMatrix4f transform = drjit::diag(ScalarVector4f(cotangent, cotangent, far_clip * reciprocal_depth, 0.f));
    transform(2, 3) = -near_clip * far_clip * reciprocal_depth;
    transform(3, 2) = 1.f;
    return transform;
}

/// Perspective projection from pinhole-camera intrinsics (focal lengths and principal point).
inline ScalarMatrix4f perspective_intrinsic(float fx, float fy, float cx, float cy,
                                            float near_clip, float far_clip) {
    const float reciprocal_depth = 1.f / (far_clip - near_clip);

    ScalarMatrix4f transform = drjit::diag(ScalarVector4f(1.f, 1.f, far_clip * reciprocal_depth, 0.f));
    transform(2, 3) = -near_clip * far_clip * reciprocal_depth;
    transform(3, 2) = 1.f;

    return translate(ScalarVector3f(1.f - 2.f * cx, 1.f - 2.f * cy, 0.f)) *
           scale(ScalarVector3f(2.f * fx, 2.f * fy, 1.f)) *
           transform;
}

/// Orthographic projection mapping z from [near_clip, far_clip] into [0, 1].
inline ScalarMatrix4f orthographic(float near_clip, float far_clip) {
    return scale(drjit::Array<float, 3>(1.f, 1.f, 1.f / (far_clip - near_clip))) *
           translate(drjit::Array<float, 3>(0.f, 0.f, -near_clip));
}

/// Camera-to-world matrix placing the camera at \p origin looking toward \p target,
/// with \p up defining the camera roll.
template <typename Float_>
drjit::Matrix<Float_, 4> look_at(const drjit::Array<Float_, 3> &origin,
                                const drjit::Array<Float_, 3> &target,
                                const drjit::Array<Float_, 3> &up) {
    const drjit::Array<Float_, 3> direction = drjit::normalize(target - origin);
    const drjit::Array<Float_, 3> left = drjit::normalize(drjit::cross(up, direction));
    const drjit::Array<Float_, 3> new_up = drjit::cross(direction, left);
    const drjit::Array<Float_, 1> z(0);

    return drjit::transpose(drjit::Matrix<Float_, 4>(
        drjit::concat(left, z),
        drjit::concat(new_up, z),
        drjit::concat(direction, z),
        drjit::Array<Float_, 4>(origin[0], origin[1], origin[2], 1.f)
    ));
}

} // namespace transform

/// Transform a position by a 4x4 matrix (w = 1) and perspective-divide back to 3D.
template <typename Float_>
drjit::Array<Float_, 3> transform_pos(const drjit::Matrix<Float_, 4> &matrix,
                                     const drjit::Array<Float_, 3> &vector) {
    const drjit::Array<Float_, 4> transformed = matrix * drjit::concat(vector, 1.f);
    return drjit::head<3>(transformed) / transformed.w();
}

/// Transform a direction by a 4x4 matrix (w = 0); ignores translation, no normalization.
template <typename Float_>
drjit::Array<Float_, 3> transform_dir(const drjit::Matrix<Float_, 4> &matrix,
                                     const drjit::Array<Float_, 3> &vector) {
    return drjit::head<3>(matrix * drjit::concat(vector, 0.f));
}

/// Transform a 2D position by a 3x3 matrix (w = 1) and perspective-divide back to 2D.
template <typename Float_>
drjit::Array<Float_, 2> transform2d_pos(const drjit::Matrix<Float_, 3> &matrix,
                                       const drjit::Array<Float_, 2> &vector) {
    const drjit::Array<Float_, 3> transformed = matrix * drjit::Array<Float_, 3>(vector[0], vector[1], 1.f);
    return drjit::head<2>(transformed) / transformed.z();
}

} // namespace rayd
