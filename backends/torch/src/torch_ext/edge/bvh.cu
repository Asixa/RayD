#include <rayd/torch/edge/bvh.h>
#include <rayd/torch/common/optix_context.h>
#include <rayd/shared/edge/edge_aabb.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace rayd::torch_backend {

[[noreturn]] inline void throw_runtime_error_local(const std::string &message) {
    throw std::runtime_error(message);
}

inline void require_local(bool condition, const std::string &message) {
    if (!condition)
        throw_runtime_error_local(message);
}

void compute_edge_optix_aabbs_cuda(
    int64_t edge_count,
    const at::Tensor &edge_p0_x,
    const at::Tensor &edge_p0_y,
    const at::Tensor &edge_p0_z,
    const at::Tensor &edge_e1_x,
    const at::Tensor &edge_e1_y,
    const at::Tensor &edge_e1_z,
    float radius,
    at::Tensor &out_aabbs) {
    require_local(edge_count >= 0, "compute_edge_optix_aabbs_cuda(): edge_count must be non-negative.");
    if (edge_count > static_cast<int64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("compute_edge_optix_aabbs_cuda(): edge_count exceeds int32 range.");
    if (edge_count == 0)
        return;
    require_local(edge_p0_x.data_ptr<float>() != nullptr &&
                      edge_p0_y.data_ptr<float>() != nullptr &&
                      edge_p0_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge start pointer is null.");
    require_local(edge_e1_x.data_ptr<float>() != nullptr &&
                      edge_e1_y.data_ptr<float>() != nullptr &&
                      edge_e1_z.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): edge vector pointer is null.");
    require_local(out_aabbs.data_ptr<float>() != nullptr,
                  "compute_edge_optix_aabbs_cuda(): output pointer is null.");

    TorchCudaContext torch_ctx = current_torch_cuda_context();
    rayd::shared::edge::launch_edge_aabb(
        static_cast<int>(edge_count),
        edge_p0_x.data_ptr<float>(),
        edge_p0_y.data_ptr<float>(),
        edge_p0_z.data_ptr<float>(),
        edge_e1_x.data_ptr<float>(),
        edge_e1_y.data_ptr<float>(),
        edge_e1_z.data_ptr<float>(),
        radius,
        out_aabbs.data_ptr<float>(),
        torch_ctx.stream);
}

} // namespace rayd::torch_backend
