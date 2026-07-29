// Copyright Xingyu Chen.
// Implements transmission support for transmission.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <rayd/transmission/layer_stack.cuh>
#include <src/bindings/tensor_contract.h>
#include <src/field_transport/ad.cuh>
#include <rayd/transmission.h>

// Debug/parity surface for the shared em/ layer-stack core: evaluates the
// full stack r/t (both polarizations) plus power R/T per input angle. This is
// the oracle-parity op the CPU complex128 golden tests compare against.

namespace {

constexpr int kBlockSize = 256;
namespace em = rayd::shared::transmission;
namespace utd = ::rayd::shared::diffraction;

__global__ void em_layer_stack_eval_kernel(int64_t count, const float* cos_theta, const int* material_id,
                                           const int* layer_offset, const int* layer_count,
                                           const float* layer_thickness_m, const float* layer_eps_r,
                                           const float* layer_sigma_e, const float* layer_mu_r, int64_t material_count,
                                           float frequency_hz, float* r_te_real, float* r_te_imag, float* r_tm_real,
                                           float* r_tm_imag, float* t_te_real, float* t_te_imag, float* t_tm_real,
                                           float* t_tm_imag, float* cap_r_te, float* cap_r_tm, float* cap_t_te,
                                           float* cap_t_tm) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            r_te_real[index] = 0.0f;
            r_te_imag[index] = 0.0f;
            r_tm_real[index] = 0.0f;
            r_tm_imag[index] = 0.0f;
            t_te_real[index] = 0.0f;
            t_te_imag[index] = 0.0f;
            t_tm_real[index] = 0.0f;
            t_tm_imag[index] = 0.0f;
            cap_r_te[index] = 0.0f;
            cap_r_tm[index] = 0.0f;
            cap_t_te[index] = 0.0f;
            cap_t_tm[index] = 0.0f;
            continue;
        }
        em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, material,
        };
        const em::StackRT te = em::stack_rt(cos_theta[index], layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(cos_theta[index], layers, frequency_hz, em::kPolTM);
        r_te_real[index] = te.r.re;
        r_te_imag[index] = te.r.im;
        r_tm_real[index] = tm.r.re;
        r_tm_imag[index] = tm.r.im;
        t_te_real[index] = te.t.re;
        t_te_imag[index] = te.t.im;
        t_tm_real[index] = tm.t.re;
        t_tm_imag[index] = tm.t.im;
        cap_r_te[index] = te.cap_r;
        cap_r_tm[index] = tm.cap_r;
        cap_t_te[index] = te.cap_t;
        cap_t_tm[index] = tm.cap_t;
    }
}

int launch_blocks(int64_t count) {
    return static_cast<int>((count + kBlockSize - 1) / kBlockSize);
}

// ---------------------------------------------------------------------------
// Backward / JVP companions of em_layer_stack_eval_kernel (plan 07 AD-3, MC
// transmission radiomap). Differentiable inputs: cos_theta (per row), the CSR
// layer thickness / eps_r / sigma_e, and the carrier frequency. layer_mu_r,
// the material ids and the CSR topology stay fixed. Derivatives come from
// stack_rt_dual, which mirrors em::stack_rt clamp for clamp; invalid material
// rows produced zero outputs in the forward and carry zero derivatives here.
// ---------------------------------------------------------------------------

namespace ad = rayd::torch::field_transport_ad;

struct StackZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct StackBasisSeed {
    int slot;
    int param; // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct StackTangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {t_thickness != nullptr ? t_thickness[query] : 0.0f, t_eps != nullptr ? t_eps[query] : 0.0f,
                t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

// Kernel-argument bundles for the twelve output arrays (passed by value so
// the pointers live in kernel parameter space, not host memory).
struct StackGradPtrs {
    const float* p[12];
};

struct StackTangentPtrs {
    float* p[12];
};

// Fold the twelve per-output tangents of one (te, tm) dual evaluation against
// the row's cotangents (output order matches em_layer_stack_eval).
__device__ __forceinline__ float stack_adj_combine(const ad::DualStackRT& te, const ad::DualStackRT& tm,
                                                   const float g[12]) {
    return g[0] * te.r.d.re + g[1] * te.r.d.im + g[2] * tm.r.d.re + g[3] * tm.r.d.im + g[4] * te.t.d.re +
           g[5] * te.t.d.im + g[6] * tm.t.d.re + g[7] * tm.t.d.im + g[8] * te.cap_r.d + g[9] * tm.cap_r.d +
           g[10] * te.cap_t.d + g[11] * tm.cap_t.d;
}

__global__ void em_layer_stack_backward_kernel(int64_t count, const float* cos_theta, const int* material_id,
                                               const int* layer_offset, const int* layer_count,
                                               const float* layer_thickness_m, const float* layer_eps_r,
                                               const float* layer_sigma_e, const float* layer_mu_r,
                                               int64_t material_count, float frequency_hz,
                                               StackGradPtrs grad_outputs, // 12 nullable cotangent arrays
                                               float* grad_cos_theta, float* grad_layer_thickness,
                                               float* grad_layer_eps_r, float* grad_layer_sigma_e,
                                               float* grad_frequency) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count)
            continue;
        float g[12];
        bool any = false;
        for (int field = 0; field < 12; ++field) {
            g[field] = grad_outputs.p[field] != nullptr ? grad_outputs.p[field][index] : 0.0f;
            any = any || g[field] != 0.0f;
        }
        if (!any)
            continue;
        em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, material,
        };
        const float ct = cos_theta[index];
        const StackZeroSeed zero_seed;
        if (grad_frequency != nullptr) {
            const ad::DualStackRT te = ad::stack_rt_dual(ct, layers, frequency_hz, 0.0f, 1.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(ct, layers, frequency_hz, 0.0f, 1.0f, em::kPolTM, zero_seed);
            atomicAdd(grad_frequency, stack_adj_combine(te, tm, g));
        }
        if (grad_cos_theta != nullptr) {
            const ad::DualStackRT te = ad::stack_rt_dual(ct, layers, frequency_hz, 1.0f, 0.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(ct, layers, frequency_hz, 1.0f, 0.0f, em::kPolTM, zero_seed);
            grad_cos_theta[index] = stack_adj_combine(te, tm, g);
        }
        if (grad_layer_thickness != nullptr || grad_layer_eps_r != nullptr || grad_layer_sigma_e != nullptr) {
            const int first = layer_offset[material];
            const int layers_in_material = layer_count[material];
            for (int layer = 0; layer < layers_in_material; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* destination = param == 0   ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    if (destination == nullptr)
                        continue;
                    const StackBasisSeed seed{slot, param};
                    const ad::DualStackRT te =
                        ad::stack_rt_dual(ct, layers, frequency_hz, 0.0f, 0.0f, em::kPolTE, seed);
                    const ad::DualStackRT tm =
                        ad::stack_rt_dual(ct, layers, frequency_hz, 0.0f, 0.0f, em::kPolTM, seed);
                    atomicAdd(destination + slot, stack_adj_combine(te, tm, g));
                }
            }
        }
    }
}

__global__ void em_layer_stack_jvp_kernel(int64_t count, const float* cos_theta, const int* material_id,
                                          const int* layer_offset, const int* layer_count,
                                          const float* layer_thickness_m, const float* layer_eps_r,
                                          const float* layer_sigma_e, const float* layer_mu_r, int64_t material_count,
                                          float frequency_hz, const float* tangent_cos_theta,
                                          const float* tangent_layer_thickness, const float* tangent_layer_eps_r,
                                          const float* tangent_layer_sigma_e, float tangent_frequency,
                                          StackTangentPtrs output_tangents) { // 12 tangent arrays
    const StackTangentSeed seed{tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int material = material_id[index];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            for (int field = 0; field < 12; ++field)
                output_tangents.p[field][index] = 0.0f;
            continue;
        }
        em::LayerView layers{
            layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, material,
        };
        const float d_cos = tangent_cos_theta != nullptr ? tangent_cos_theta[index] : 0.0f;
        const ad::DualStackRT te =
            ad::stack_rt_dual(cos_theta[index], layers, frequency_hz, d_cos, tangent_frequency, em::kPolTE, seed);
        const ad::DualStackRT tm =
            ad::stack_rt_dual(cos_theta[index], layers, frequency_hz, d_cos, tangent_frequency, em::kPolTM, seed);
        output_tangents.p[0][index] = te.r.d.re;
        output_tangents.p[1][index] = te.r.d.im;
        output_tangents.p[2][index] = tm.r.d.re;
        output_tangents.p[3][index] = tm.r.d.im;
        output_tangents.p[4][index] = te.t.d.re;
        output_tangents.p[5][index] = te.t.d.im;
        output_tangents.p[6][index] = tm.t.d.re;
        output_tangents.p[7][index] = tm.t.d.im;
        output_tangents.p[8][index] = te.cap_r.d;
        output_tangents.p[9][index] = tm.cap_r.d;
        output_tangents.p[10][index] = te.cap_t.d;
        output_tangents.p[11][index] = tm.cap_t.d;
    }
}

} // namespace

rayd::torch::LayerStackResult rayd::torch::em_layer_stack_eval(const rayd::torch::LayerStackRequest& request) {
    const auto& cos_theta = request.cos_theta;
    const auto& material_id = request.material_id;
    const auto& layer_offset = request.layer_offset;
    const auto& layer_count = request.layer_count;
    const auto& layer_thickness_m = request.layer_thickness_m;
    const auto& layer_eps_r = request.layer_eps_r;
    const auto& layer_sigma_e = request.layer_sigma_e;
    const auto& layer_mu_r = request.layer_mu_r;
    const double frequency_hz = request.frequency_hz;
    auto check_flat_tensor = [](const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
        rayd::torch_backend::require_cuda(tensor, name);
        rayd::torch_backend::require_dtype(tensor, dtype, name);
        rayd::torch_backend::require_rank(tensor, 1, name);
        rayd::torch_backend::require_contiguous(tensor, name);
    };
    check_flat_tensor(cos_theta, "cos_theta", at::kFloat);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(material_id.size(0) == count, "material_id must match cos_theta rows");
    TORCH_CHECK(layer_count.size(0) == material_count, "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total, "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor :
         {material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.get_device() == cos_theta.get_device(),
                    "em_layer_stack_eval tensors must share one CUDA device");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
    const c10::cuda::CUDAGuard guard(static_cast<int>(cos_theta.get_device()));

    auto options = cos_theta.options();
    at::Tensor outputs[12];
    for (int field = 0; field < 12; ++field)
        outputs[field] = at::empty({count}, options);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_eval_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, cos_theta.data_ptr<float>(), material_id.data_ptr<int>(), layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(), layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(), material_count,
            static_cast<float>(frequency_hz), outputs[0].data_ptr<float>(), outputs[1].data_ptr<float>(),
            outputs[2].data_ptr<float>(), outputs[3].data_ptr<float>(), outputs[4].data_ptr<float>(),
            outputs[5].data_ptr<float>(), outputs[6].data_ptr<float>(), outputs[7].data_ptr<float>(),
            outputs[8].data_ptr<float>(), outputs[9].data_ptr<float>(), outputs[10].data_ptr<float>(),
            outputs[11].data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4],  outputs[5],
            outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11]};
}

namespace {

constexpr const char* kStackFields[12] = {
    "r_te_real", "r_te_imag", "r_tm_real", "r_tm_imag", "t_te_real", "t_te_imag",
    "t_tm_real", "t_tm_imag", "cap_R_te",  "cap_R_tm",  "cap_T_te",  "cap_T_tm",
};

at::Tensor stack_zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(
            cudaMemsetAsync(tensor.data_ptr(), 0, static_cast<size_t>(tensor.numel()) * tensor.element_size(), stream));
    }
    return tensor;
}

void check_stack_primal(const at::Tensor& cos_theta, const at::Tensor& material_id, const at::Tensor& layer_offset,
                        const at::Tensor& layer_count, const at::Tensor& layer_thickness_m,
                        const at::Tensor& layer_eps_r, const at::Tensor& layer_sigma_e, const at::Tensor& layer_mu_r,
                        double frequency_hz) {
    auto check_flat_tensor = [](const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
        rayd::torch_backend::require_cuda(tensor, name);
        rayd::torch_backend::require_dtype(tensor, dtype, name);
        rayd::torch_backend::require_rank(tensor, 1, name);
        rayd::torch_backend::require_contiguous(tensor, name);
    };
    check_flat_tensor(cos_theta, "cos_theta", at::kFloat);
    check_flat_tensor(material_id, "material_id", at::kInt);
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);
    TORCH_CHECK(material_id.size(0) == cos_theta.size(0), "material_id must match cos_theta rows");
    TORCH_CHECK(layer_count.size(0) == layer_offset.size(0), "layer_count must match layer_offset rows");
    const int64_t layer_total = layer_thickness_m.size(0);
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total, "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor :
         {material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.get_device() == cos_theta.get_device(), "em_layer_stack tensors must share one CUDA device");
    TORCH_CHECK(frequency_hz > 0.0, "frequency_hz must be positive");
}

const float* stack_optional_grad(const std::optional<at::Tensor>& value, const char* name, int64_t rows,
                                 const at::Tensor& reference) {
    if (!value.has_value())
        return nullptr;
    const at::Tensor& storage = *value;
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(storage.dim() == 1 && storage.size(0) == rows, name, " must have one value per row");
    TORCH_CHECK(storage.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(storage.get_device() == reference.get_device(), name, " must share the primal device");
    return storage.data_ptr<float>();
}

} // namespace

rayd::torch::LayerStackBackwardResult rayd::torch::em_layer_stack_backward(
    const rayd::torch::LayerStackBackwardRequest& request) {
    const auto& cos_theta = request.primal.cos_theta;
    const auto& material_id = request.primal.material_id;
    const auto& layer_offset = request.primal.layer_offset;
    const auto& layer_count = request.primal.layer_count;
    const auto& layer_thickness_m = request.primal.layer_thickness_m;
    const auto& layer_eps_r = request.primal.layer_eps_r;
    const auto& layer_sigma_e = request.primal.layer_sigma_e;
    const auto& layer_mu_r = request.primal.layer_mu_r;
    const double frequency_hz = request.primal.frequency_hz;
    const bool need_cos_theta = request.need_cos_theta;
    const bool need_layers = request.need_layers;
    const bool need_frequency = request.need_frequency;
    check_stack_primal(cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
                       layer_mu_r, frequency_hz);
    const c10::cuda::CUDAGuard guard(static_cast<int>(cos_theta.get_device()));
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    StackGradPtrs grads{};
    for (int field = 0; field < 12; ++field) {
        grads.p[field] = stack_optional_grad(request.grad_outputs[field], kStackFields[field], count, cos_theta);
    }

    auto options = cos_theta.options();
    auto grad_cos_theta = stack_zero_filled({count}, options);
    auto grad_layer_thickness = stack_zero_filled({layer_total}, options);
    auto grad_layer_eps_r = stack_zero_filled({layer_total}, options);
    auto grad_layer_sigma_e = stack_zero_filled({layer_total}, options);
    auto grad_frequency = stack_zero_filled({1}, options);
    if (count > 0 && (need_cos_theta || need_layers || need_frequency)) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, cos_theta.data_ptr<float>(), material_id.data_ptr<int>(), layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(), layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(), material_count,
            static_cast<float>(frequency_hz), grads, need_cos_theta ? grad_cos_theta.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_thickness.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_eps_r.data_ptr<float>() : nullptr,
            need_layers ? grad_layer_sigma_e.data_ptr<float>() : nullptr,
            need_frequency ? grad_frequency.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_cos_theta, grad_layer_thickness, grad_layer_eps_r, grad_layer_sigma_e, grad_frequency};
}

rayd::torch::LayerStackResult rayd::torch::em_layer_stack_jvp(const rayd::torch::LayerStackJvpRequest& request) {
    const auto& cos_theta = request.primal.cos_theta;
    const auto& material_id = request.primal.material_id;
    const auto& layer_offset = request.primal.layer_offset;
    const auto& layer_count = request.primal.layer_count;
    const auto& layer_thickness_m = request.primal.layer_thickness_m;
    const auto& layer_eps_r = request.primal.layer_eps_r;
    const auto& layer_sigma_e = request.primal.layer_sigma_e;
    const auto& layer_mu_r = request.primal.layer_mu_r;
    const double frequency_hz = request.primal.frequency_hz;
    const double tangent_frequency = request.tangent_frequency;
    check_stack_primal(cos_theta, material_id, layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e,
                       layer_mu_r, frequency_hz);
    const c10::cuda::CUDAGuard guard(static_cast<int>(cos_theta.get_device()));
    const int64_t count = cos_theta.size(0);
    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);

    const float* t_cos = stack_optional_grad(request.tangent_cos_theta, "tangent_cos_theta", count, cos_theta);
    const float* t_thickness =
        stack_optional_grad(request.tangent_layer_thickness_m, "tangent_layer_thickness_m", layer_total, cos_theta);
    const float* t_eps =
        stack_optional_grad(request.tangent_layer_eps_r, "tangent_layer_eps_r", layer_total, cos_theta);
    const float* t_sigma =
        stack_optional_grad(request.tangent_layer_sigma_e, "tangent_layer_sigma_e", layer_total, cos_theta);

    auto options = cos_theta.options();
    at::Tensor outputs[12];
    StackTangentPtrs tangents{};
    for (int field = 0; field < 12; ++field) {
        outputs[field] = at::empty({count}, options);
        tangents.p[field] = outputs[field].data_ptr<float>();
    }
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(cos_theta.get_device()).stream();
        em_layer_stack_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, cos_theta.data_ptr<float>(), material_id.data_ptr<int>(), layer_offset.data_ptr<int>(),
            layer_count.data_ptr<int>(), layer_thickness_m.data_ptr<float>(), layer_eps_r.data_ptr<float>(),
            layer_sigma_e.data_ptr<float>(), layer_mu_r.data_ptr<float>(), material_count,
            static_cast<float>(frequency_hz), t_cos, t_thickness, t_eps, t_sigma, static_cast<float>(tangent_frequency),
            tangents);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4],  outputs[5],
            outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11]};
}

// ---- merged from src/transmission/sequence_part.cu ----

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <rayd/field_transport.cuh>
#include <rayd/transmission/layer_stack.cuh>
#include <src/bindings/tensor_contract.h>
#include <rayd/transmission.h>

#include <utility>

namespace {

namespace field = rayd::shared::diffraction;
namespace em = rayd::shared::transmission;
namespace transport = rayd::shared::field_transport;

__device__ __forceinline__ field::float3a load3(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}
__device__ __forceinline__ field::float3a load_sequence3(const float* values, int64_t index, int64_t bounce,
                                                         int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ c10::complex<float> to_complex(field::Complex value) {
    return c10::complex<float>(value.re, value.im);
}

// Endpoint-connection specular transmission (contract section 4).
//
// The field travels along the straight source->target ray. Each wall applies
// the layer-stack Jones operator diag(t_TE, t_TM) in the wall's s/p basis;
// t_stack is interface-to-interface, so it already carries the interior
// k_z*d phase and absorption. The exterior free-space carrier phase runs
// over (L - sum_w d_w/cos(theta_w)) and each wall adds the lateral chord
// phase exp(-j*k_par*d_w*tan(theta_w)) with k_par = k0*sin(theta_w). Both are
// pure k0 phases, so they collapse to one carrier over the effective length
//   L_eff = L - sum_w d_w/cos(theta_w) + sum_w d_w*sin^2(theta_w)/cos(theta_w)
//         = L - sum_w d_w*cos(theta_w),
// which is what this kernel accumulates. Amplitude spreading uses the FULL
// straight length (1/(2*k*L), matching free_space_complex3), and
// path_length_m/delay_s report the full straight length.
//
// Vacuum-wall identity: for a single vacuum layer t = exp(-j*k0*cos(theta)*d)
// and L_eff = L - d*cos(theta), so t * exp(-j*k0*L_eff) = exp(-j*k0*L) and
// the output equals the free-space field with no wall (unit tested).
__global__ void transmission_sequence_kernel(
    int64_t count, int64_t depth, const bool* path_valid, const float* source, const float* target,
    const float* interaction_normals, const int* interaction_material_id, const bool* interaction_valid,
    const float* tx_power, const float* tx_polarization, const float* rx_polarization, const int* layer_offset,
    const int* layer_count, const float* layer_thickness_m, const float* layer_eps_r, const float* layer_sigma_e,
    const float* layer_mu_r, int64_t material_count, float frequency_hz, c10::complex<float>* field_vector,
    c10::complex<float>* coefficient, c10::complex<float>* path_field, float* path_gain, float* path_length,
    float* delay, float* direction_out) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        if (!path_valid[index]) {
            field_vector[base] = c10::complex<float>(0.0F, 0.0F);
            field_vector[base + 1] = c10::complex<float>(0.0F, 0.0F);
            field_vector[base + 2] = c10::complex<float>(0.0F, 0.0F);
            coefficient[index] = c10::complex<float>(0.0F, 0.0F);
            path_field[index] = c10::complex<float>(0.0F, 0.0F);
            path_gain[index] = 0.0F;
            path_length[index] = 0.0F;
            delay[index] = 0.0F;
            direction_out[base] = 0.0F;
            direction_out[base + 1] = 0.0F;
            direction_out[base + 2] = 0.0F;
            continue;
        }
        const field::float3a source_value = load3(source, index);
        const field::float3a target_value = load3(target, index);
        const field::float3a offset = field::f3_sub(target_value, source_value);
        const float total_length = field::safe_length(offset);
        const field::float3a direction = field::safe_normalize(offset, field::make_f3(0.0f, 0.0f, 1.0f));
        // F1: unnormalized transverse projection of the transmit polarization.
        const field::float3a tx_axis = field::project_to_wedge_plane(load3(tx_polarization, index), direction);
        field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
        float carrier_length = total_length;
        bool chain_valid = true;
        for (int64_t wall = 0; wall < depth; ++wall) {
            const int64_t scalar = index * depth + wall;
            if (!interaction_valid[scalar])
                continue;
            const int material = interaction_material_id[scalar];
            if (material < 0 || static_cast<int64_t>(material) >= material_count) {
                chain_valid = false;
                break;
            }
            // s/p basis of the wall; outgoing direction equals incident
            // direction, so the incident basis is also the exit basis. The
            // alternate axis keeps the basis deterministic at normal
            // incidence (same construction as reflect_complex3).
            const transport::WallFrame frame =
                transport::wall_frame(direction, load_sequence3(interaction_normals, index, wall, depth));
            em::LayerView layers{
                layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, material,
            };
            const em::StackRT te = em::stack_rt(frame.cos_theta, layers, frequency_hz, em::kPolTE);
            const em::StackRT tm = em::stack_rt(frame.cos_theta, layers, frequency_hz, em::kPolTM);
            const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
            const field::Complex e_p = transport::complex3_dot_real(value, frame.p_axis);
            value = field::c3_add(field::cplx_scale_real(frame.s_axis, field::cplx_mul(te.t, e_s)),
                                  field::cplx_scale_real(frame.p_axis, field::cplx_mul(tm.t, e_p)));
            float wall_thickness = 0.0f;
            const int first = layer_offset[material];
            const int layers_in_wall = layer_count[material];
            for (int layer = 0; layer < layers_in_wall; ++layer)
                wall_thickness += fmaxf(layer_thickness_m[first + layer], 0.0f);
            carrier_length -= wall_thickness * frame.cos_theta;
        }
        const float wave_number = 2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
        const float amplitude = 1.0f / (2.0f * wave_number * fmaxf(total_length, field::UTD_EPS));
        const field::Complex propagation =
            field::cplx_mul_real(field::cplx_exp_phase(transport::precise_neg_kd(wave_number, carrier_length)),
                                 amplitude);
        value = field::c3_scale(value, propagation);
        if (!chain_valid)
            value = field::c3_zero();
        field_vector[base] = to_complex(value.x);
        field_vector[base + 1] = to_complex(value.y);
        field_vector[base + 2] = to_complex(value.z);
        const field::Complex scalar_field =
            transport::project_receiver(value, direction, load3(rx_polarization, index));
        coefficient[index] = to_complex(scalar_field);
        const field::Complex received = field::cplx_mul_real(scalar_field, sqrtf(fmaxf(tx_power[index], 0.0f)));
        path_field[index] = to_complex(received);
        path_gain[index] = field::cplx_abs_sqr(received);
        path_length[index] = total_length;
        delay[index] = total_length / transport::kSpeedOfLight;
        direction_out[base] = direction.x;
        direction_out[base + 1] = direction.y;
        direction_out[base + 2] = direction.z;
    }
}

void check_tensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank) {
    rayd::torch_backend::require_cuda(tensor, name);
    rayd::torch_backend::require_dtype(tensor, dtype, name);
    rayd::torch_backend::require_rank(tensor, rank, name);
    rayd::torch_backend::require_contiguous(tensor, name);
}

void check_vec3_table(const at::Tensor& tensor, const char* name) {
    check_tensor(tensor, name, at::kFloat, 2);
    TORCH_CHECK(tensor.size(1) == 3, name, " must have shape (N, 3)");
}

void check_flat_tensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype) {
    check_tensor(tensor, name, dtype, 1);
}

std::pair<int64_t, int64_t> check_transmission_primal(const rayd::torch::TransmissionSequenceRequest& request) {
    check_vec3_table(request.source, "source");
    check_flat_tensor(request.path_valid, "path_valid", at::kBool);
    check_vec3_table(request.target, "target");
    check_tensor(request.interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(request.interaction_normals, "interaction_normals", at::kFloat, 3);
    check_tensor(request.interaction_material_id, "interaction_material_id", at::kInt, 2);
    check_tensor(request.interaction_valid, "interaction_valid", at::kBool, 2);
    check_flat_tensor(request.tx_power, "tx_power", at::kFloat);
    check_vec3_table(request.tx_polarization, "tx_polarization");
    check_vec3_table(request.rx_polarization, "rx_polarization");
    check_flat_tensor(request.layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(request.layer_count, "layer_count", at::kInt);
    check_flat_tensor(request.layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(request.layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(request.layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(request.layer_mu_r, "layer_mu_r", at::kFloat);

    const int64_t count = request.source.size(0);
    const int64_t depth = request.interaction_positions.size(1);
    TORCH_CHECK(depth > 0 && request.interaction_positions.size(2) == 3,
                "interaction_positions must have shape (N, D, 3) with D > 0");
    TORCH_CHECK(request.interaction_positions.size(0) == count, "interaction_positions must match source rows");
    TORCH_CHECK(request.interaction_normals.sizes() == request.interaction_positions.sizes(),
                "interaction_normals must match interaction_positions");
    TORCH_CHECK(request.interaction_material_id.size(0) == count && request.interaction_material_id.size(1) == depth,
                "interaction_material_id must have shape (N, D)");
    TORCH_CHECK(request.interaction_valid.size(0) == count && request.interaction_valid.size(1) == depth,
                "interaction_valid must have shape (N, D)");
    TORCH_CHECK(request.path_valid.size(0) == count, "path_valid must match source rows");
    TORCH_CHECK(request.target.size(0) == count && request.tx_power.size(0) == count &&
                    request.tx_polarization.size(0) == count && request.rx_polarization.size(0) == count,
                "transmission endpoint tensors must match source rows");
    const int64_t material_count = request.layer_offset.size(0);
    const int64_t layer_total = request.layer_thickness_m.size(0);
    TORCH_CHECK(request.layer_count.size(0) == material_count, "layer_count must match layer_offset rows");
    for (const auto& tensor : {request.layer_eps_r, request.layer_sigma_e, request.layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total, "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor :
         {request.path_valid, request.target, request.interaction_positions, request.interaction_normals,
          request.interaction_material_id, request.interaction_valid, request.tx_power, request.tx_polarization,
          request.rx_polarization, request.layer_offset, request.layer_count, request.layer_thickness_m,
          request.layer_eps_r, request.layer_sigma_e, request.layer_mu_r})
        TORCH_CHECK(tensor.get_device() == request.source.get_device(),
                    "transmission tensors must share one CUDA device");
    TORCH_CHECK(request.frequency_hz > 0.0, "frequency_hz must be positive");
    return {depth, material_count};
}

} // namespace

rayd::torch::TransmissionSequenceResult rayd::torch::field_transmission_sequence(
    const rayd::torch::TransmissionSequenceRequest& request) {
    const auto [depth, material_count] = check_transmission_primal(request);
    const c10::cuda::CUDAGuard guard(static_cast<int>(request.source.get_device()));
    const int64_t count = request.source.size(0);
    auto complex_options = request.source.options().dtype(at::kComplexFloat);
    auto field_vector = at::empty({count, 3}, complex_options);
    auto coefficient = at::empty({count}, complex_options);
    auto path_field = at::empty({count}, complex_options);
    auto path_gain = at::empty({count}, request.source.options());
    auto path_length = at::empty_like(path_gain);
    auto delay = at::empty_like(path_gain);
    auto direction = at::empty_like(request.source);
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(request.source.get_device()).stream();
        transmission_sequence_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, request.path_valid.data_ptr<bool>(), request.source.data_ptr<float>(),
            request.target.data_ptr<float>(), request.interaction_normals.data_ptr<float>(),
            request.interaction_material_id.data_ptr<int>(), request.interaction_valid.data_ptr<bool>(),
            request.tx_power.data_ptr<float>(), request.tx_polarization.data_ptr<float>(),
            request.rx_polarization.data_ptr<float>(), request.layer_offset.data_ptr<int>(),
            request.layer_count.data_ptr<int>(), request.layer_thickness_m.data_ptr<float>(),
            request.layer_eps_r.data_ptr<float>(), request.layer_sigma_e.data_ptr<float>(),
            request.layer_mu_r.data_ptr<float>(), material_count, static_cast<float>(request.frequency_hz),
            field_vector.data_ptr<c10::complex<float>>(), coefficient.data_ptr<c10::complex<float>>(),
            path_field.data_ptr<c10::complex<float>>(), path_gain.data_ptr<float>(), path_length.data_ptr<float>(),
            delay.data_ptr<float>(), direction.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {field_vector, coefficient, path_field, path_gain, path_length, delay, direction};
}

// ---- merged from src/transmission/sequence_ad_part.cu ----

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <rayd/field_transport.cuh>
#include <rayd/transmission/layer_stack.cuh>
#include <src/bindings/tensor_contract.h>
#include <src/field_transport/ad.cuh>
#include <rayd/transmission.h>

#include <optional>
#include <utility>

namespace {

constexpr int kMaxAdDepth = 8;
namespace field = rayd::shared::diffraction;
namespace em = rayd::shared::transmission;
namespace transport = rayd::shared::field_transport;
namespace ad = rayd::torch::field_transport_ad;

using ad::adj_dot;
using ad::DualC;
using ad::fold_output_cotangents;
using ad::write_output_tangents;

__device__ __forceinline__ field::float3a load3f(const float* values, int64_t index) {
    const int64_t base = index * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ field::float3a load_sequence3f(const float* values, int64_t index, int64_t bounce,
                                                          int64_t depth) {
    const int64_t base = (index * depth + bounce) * 3;
    return field::make_f3(values[base], values[base + 1], values[base + 2]);
}

__device__ __forceinline__ ad::DualF3 load_dual3f(const float* values, const float* tangents, int64_t index) {
    return {load3f(values, index), tangents != nullptr ? load3f(tangents, index) : field::f3_zero()};
}

__device__ __forceinline__ ad::DualF3 load_dual_sequence3f(const float* values, const float* tangents, int64_t index,
                                                           int64_t bounce, int64_t depth) {
    return {load_sequence3f(values, index, bounce, depth),
            tangents != nullptr ? load_sequence3f(tangents, index, bounce, depth) : field::f3_zero()};
}

struct TransmissionChain {
    transport::WallFrame frames[kMaxAdDepth];
    field::Complex3 value_in[kMaxAdDepth]; // field entering each wall
    field::Complex e_s[kMaxAdDepth];
    field::Complex e_p[kMaxAdDepth];
    field::Complex t_te[kMaxAdDepth];
    field::Complex t_tm[kMaxAdDepth];
    float wall_thickness[kMaxAdDepth];
    int wall_material[kMaxAdDepth]; // -1 for skipped slots
    field::float3a direction;
    field::float3a rx_axis;
    field::Complex3 value_chain;
    field::Complex propagation;
    field::Complex propagation_dfreq;
    field::Complex propagation_dcarrier;
    field::Complex propagation_dtotal; // amplitude spread over the raw length
    float total_length;
    float carrier_length;
    float amplitude_scale;
    bool path_valid;
};

__device__ void transmission_chain_eval(int64_t index, int64_t depth, const float* source, const float* target,
                                        const float* interaction_normals, const int* interaction_material_id,
                                        const bool* interaction_valid, const float* tx_power,
                                        const float* tx_polarization, const float* rx_polarization,
                                        const em::LayerView& layers_base, int64_t material_count, float frequency_hz,
                                        TransmissionChain& chain) {
    const field::float3a source_value = load3f(source, index);
    const field::float3a target_value = load3f(target, index);
    const field::float3a offset = field::f3_sub(target_value, source_value);
    chain.total_length = field::safe_length(offset);
    chain.direction = field::safe_normalize(offset, field::make_f3(0.0f, 0.0f, 1.0f));
    // F1: unnormalized transverse projection of the transmit polarization.
    const field::float3a tx_axis = field::project_to_wedge_plane(load3f(tx_polarization, index), chain.direction);
    field::Complex3 value = field::cplx_scale_real(tx_axis, field::cplx(1.0f, 0.0f));
    float carrier_length = chain.total_length;
    chain.path_valid = true;
    for (int64_t wall = 0; wall < depth; ++wall) {
        chain.wall_material[wall] = -1;
        const int64_t scalar = index * depth + wall;
        if (!interaction_valid[scalar])
            continue;
        const int material = interaction_material_id[scalar];
        if (material < 0 || static_cast<int64_t>(material) >= material_count) {
            chain.path_valid = false;
            break;
        }
        const transport::WallFrame frame =
            transport::wall_frame(chain.direction, load_sequence3f(interaction_normals, index, wall, depth));
        em::LayerView layers = layers_base;
        layers.material = material;
        const em::StackRT te = em::stack_rt(frame.cos_theta, layers, frequency_hz, em::kPolTE);
        const em::StackRT tm = em::stack_rt(frame.cos_theta, layers, frequency_hz, em::kPolTM);
        const field::Complex e_s = transport::complex3_dot_real(value, frame.s_axis);
        const field::Complex e_p = transport::complex3_dot_real(value, frame.p_axis);
        chain.frames[wall] = frame;
        chain.value_in[wall] = value;
        chain.e_s[wall] = e_s;
        chain.e_p[wall] = e_p;
        chain.t_te[wall] = te.t;
        chain.t_tm[wall] = tm.t;
        chain.wall_material[wall] = material;
        value = field::c3_add(field::cplx_scale_real(frame.s_axis, field::cplx_mul(te.t, e_s)),
                              field::cplx_scale_real(frame.p_axis, field::cplx_mul(tm.t, e_p)));
        float wall_thickness = 0.0f;
        const int first = layers_base.layer_offset[material];
        const int layers_in_wall = layers_base.layer_count[material];
        for (int layer = 0; layer < layers_in_wall; ++layer)
            wall_thickness += fmaxf(layers_base.layer_thickness_m[first + layer], 0.0f);
        chain.wall_thickness[wall] = wall_thickness;
        carrier_length -= wall_thickness * frame.cos_theta;
    }
    const float wave_number = 2.0f * field::UTD_PI * frequency_hz / transport::kSpeedOfLight;
    const float amplitude = 1.0f / (2.0f * wave_number * fmaxf(chain.total_length, field::UTD_EPS));
    const field::Complex propagation =
        field::cplx_mul_real(field::cplx_exp_phase(transport::precise_neg_kd(wave_number, carrier_length)), amplitude);
    chain.value_chain = value;
    // F1: receiver scalar = p_rx . E via the unnormalized transverse of p_rx.
    chain.rx_axis = field::project_to_wedge_plane(load3f(rx_polarization, index), chain.direction);
    chain.propagation = propagation;
    chain.carrier_length = carrier_length;
    // dP/df = P * (-1/k - j*carrier) * (2*pi/c); the amplitude spreads over
    // the full straight length (geometry, handled via dP/dtotal), the phase
    // runs over the carrier length (thickness and cos_theta dependent,
    // handled via dP/dcarrier = -j*k*P).
    const field::Complex dlog = field::cplx(-1.0f / wave_number, -carrier_length);
    chain.propagation_dfreq =
        field::cplx_mul_real(field::cplx_mul(propagation, dlog), 2.0f * field::UTD_PI / transport::kSpeedOfLight);
    chain.propagation_dcarrier = field::cplx(propagation.im * wave_number, -propagation.re * wave_number);
    const float length_gate =
        chain.total_length >= field::UTD_EPS ? 1.0f / fmaxf(chain.total_length, field::UTD_EPS) : 0.0f;
    chain.propagation_dtotal = field::cplx_mul_real(propagation, -length_gate);
    chain.amplitude_scale = sqrtf(fmaxf(tx_power[index], 0.0f));
}

struct ZeroSeed {
    __device__ ad::LayerSeed operator()(int) const { return {0.0f, 0.0f, 0.0f}; }
};

struct BasisSeed {
    int slot;
    int param; // 0 thickness, 1 eps, 2 sigma
    __device__ ad::LayerSeed operator()(int query) const {
        ad::LayerSeed seed{0.0f, 0.0f, 0.0f};
        if (query == slot) {
            if (param == 0)
                seed.d_thickness = 1.0f;
            else if (param == 1)
                seed.d_eps = 1.0f;
            else
                seed.d_sigma = 1.0f;
        }
        return seed;
    }
};

struct TangentSeed {
    const float* t_thickness;
    const float* t_eps;
    const float* t_sigma;
    __device__ ad::LayerSeed operator()(int query) const {
        return {t_thickness != nullptr ? t_thickness[query] : 0.0f, t_eps != nullptr ? t_eps[query] : 0.0f,
                t_sigma != nullptr ? t_sigma[query] : 0.0f};
    }
};

__global__ void transmission_sequence_backward_kernel(
    int64_t count, int64_t depth, const bool* path_valid, const float* source, const float* target,
    const float* interaction_normals, const int* interaction_material_id, const bool* interaction_valid,
    const float* tx_power, const float* tx_polarization, const float* rx_polarization, const int* layer_offset,
    const int* layer_count, const float* layer_thickness_m, const float* layer_eps_r, const float* layer_sigma_e,
    const float* layer_mu_r, int64_t material_count, float frequency_hz, const c10::complex<float>* grad_field_vector,
    const c10::complex<float>* grad_coefficient, const c10::complex<float>* grad_path_field,
    const float* grad_path_gain, const float* grad_path_length, const float* grad_delay, float* grad_layer_thickness,
    float* grad_layer_eps_r, float* grad_layer_sigma_e, float* grad_frequency, float* grad_source, float* grad_target,
    float* grad_normals) {
    const em::LayerView layers_base{
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, 0,
    };
    const bool need_geometry = grad_source != nullptr;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (!path_valid[index])
            continue;
        TransmissionChain chain;
        transmission_chain_eval(index, depth, source, target, interaction_normals, interaction_material_id,
                                interaction_valid, tx_power, tx_polarization, rx_polarization, layers_base,
                                material_count, frequency_hz, chain);
        if (!chain.path_valid)
            continue; // forward zeroed the outputs; every gradient is zero

        const field::Complex3 value_final = field::c3_scale(chain.value_chain, chain.propagation);
        const field::Complex scalar = transport::complex3_dot_real(value_final, chain.rx_axis);
        const field::Complex path_field_value = field::cplx_mul_real(scalar, chain.amplitude_scale);
        field::Complex g_scalar = field::cplx_zero();
        field::Complex3 g_value =
            fold_output_cotangents(grad_field_vector, grad_coefficient, grad_path_field, grad_path_gain, index,
                                   chain.rx_axis, path_field_value, chain.amplitude_scale, g_scalar);

        field::Complex g_propagation = field::cplx_zero();
        field::Complex3 g_chain = field::c3_zero();
        field::adj_cplx_mul(chain.value_chain.x, chain.propagation, g_value.x, g_chain.x, g_propagation);
        field::adj_cplx_mul(chain.value_chain.y, chain.propagation, g_value.y, g_chain.y, g_propagation);
        field::adj_cplx_mul(chain.value_chain.z, chain.propagation, g_value.z, g_chain.z, g_propagation);
        float g_freq = adj_dot(g_propagation, chain.propagation_dfreq);
        const float g_carrier = adj_dot(g_propagation, chain.propagation_dcarrier);

        // Geometry cotangents (plan 07 AD-2). The straight length L feeds the
        // amplitude spread, the carrier start (carrier = L - sum_w d_w*cos_w)
        // and the path_length_m / delay_s outputs; the shared ray direction
        // feeds the tx/rx bases and every wall frame.
        field::float3a g_direction = field::f3_zero();
        float g_total_length = 0.0f;
        if (need_geometry) {
            g_total_length = adj_dot(g_propagation, chain.propagation_dtotal) + g_carrier;
            if (grad_path_length != nullptr)
                g_total_length += grad_path_length[index];
            if (grad_delay != nullptr)
                g_total_length += grad_delay[index] / transport::kSpeedOfLight;
            const field::float3a g_rx_axis = field::make_f3(field::cplx_adj_dot(g_scalar, value_final.x),
                                                            field::cplx_adj_dot(g_scalar, value_final.y),
                                                            field::cplx_adj_dot(g_scalar, value_final.z));
            field::float3a g_pol_dump = field::f3_zero();
            ad::adj_transverse_project(chain.direction, load3f(rx_polarization, index), g_rx_axis, g_direction,
                                       g_pol_dump);
        }

        for (int64_t wall = depth - 1; wall >= 0; --wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const transport::WallFrame& frame = chain.frames[wall];
            field::float3a g_s_axis = field::f3_zero();
            field::float3a g_p_axis = field::f3_zero();
            field::Complex gs = field::cplx_zero();
            field::Complex gp = field::cplx_zero();
            field::adj_cplx_scale_real(frame.s_axis, field::cplx_mul(chain.t_te[wall], chain.e_s[wall]), g_chain,
                                       g_s_axis, gs);
            field::adj_cplx_scale_real(frame.p_axis, field::cplx_mul(chain.t_tm[wall], chain.e_p[wall]), g_chain,
                                       g_p_axis, gp);
            field::Complex g_t_te = field::cplx_zero();
            field::Complex g_t_tm = field::cplx_zero();
            field::Complex g_e_s = field::cplx_zero();
            field::Complex g_e_p = field::cplx_zero();
            field::adj_cplx_mul(chain.t_te[wall], chain.e_s[wall], gs, g_t_te, g_e_s);
            field::adj_cplx_mul(chain.t_tm[wall], chain.e_p[wall], gp, g_t_tm, g_e_p);
            field::Complex3 g_value_in = field::c3_zero();
            field::adj_cplx_dot_real(chain.value_in[wall], frame.s_axis, g_e_s, g_value_in, g_s_axis);
            field::adj_cplx_dot_real(chain.value_in[wall], frame.p_axis, g_e_p, g_value_in, g_p_axis);
            g_chain = g_value_in;

            em::LayerView layers = layers_base;
            layers.material = material;
            const int first = layer_offset[material];
            const int layers_in_wall = layer_count[material];
            for (int layer = 0; layer < layers_in_wall; ++layer) {
                const int slot = first + layer;
                for (int param = 0; param < 3; ++param) {
                    float* destination = param == 0   ? grad_layer_thickness
                                         : param == 1 ? grad_layer_eps_r
                                                      : grad_layer_sigma_e;
                    if (destination == nullptr)
                        continue;
                    const BasisSeed seed{slot, param};
                    const ad::DualStackRT te =
                        ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f, em::kPolTE, seed);
                    const ad::DualStackRT tm =
                        ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 0.0f, 0.0f, em::kPolTM, seed);
                    float grad = adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
                    if (param == 0 && layer_thickness_m[slot] >= 0.0f) {
                        // Carrier phase runs over L - sum_w d_w * cos(theta_w).
                        grad += g_carrier * (-frame.cos_theta);
                    }
                    atomicAdd(destination + slot, grad);
                }
            }
            if (grad_frequency != nullptr) {
                const ZeroSeed zero_seed;
                const ad::DualStackRT te =
                    ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f, em::kPolTE, zero_seed);
                const ad::DualStackRT tm =
                    ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 0.0f, 1.0f, em::kPolTM, zero_seed);
                g_freq += adj_dot(g_t_te, te.t.d) + adj_dot(g_t_tm, tm.t.d);
            }
            if (!need_geometry)
                continue;

            // Geometry enters this wall through cos_theta (Fresnel stack and
            // carrier chord) and through the s/p frame.
            const ZeroSeed zero_seed;
            const ad::DualStackRT te_cos =
                ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f, em::kPolTE, zero_seed);
            const ad::DualStackRT tm_cos =
                ad::stack_rt_dual(frame.cos_theta, layers, frequency_hz, 1.0f, 0.0f, em::kPolTM, zero_seed);
            float g_cos_theta = adj_dot(g_t_te, te_cos.t.d) + adj_dot(g_t_tm, tm_cos.t.d);
            g_cos_theta += g_carrier * (-chain.wall_thickness[wall]);
            field::float3a g_normal_raw = field::f3_zero();
            ad::adj_wall_frame(chain.direction, load_sequence3f(interaction_normals, index, wall, depth), g_s_axis,
                               g_p_axis, g_cos_theta, g_direction, g_normal_raw);
            const int64_t normal_base = (index * depth + wall) * 3;
            grad_normals[normal_base] = g_normal_raw.x;
            grad_normals[normal_base + 1] = g_normal_raw.y;
            grad_normals[normal_base + 2] = g_normal_raw.z;
        }
        if (grad_frequency != nullptr)
            atomicAdd(grad_frequency, g_freq);
        if (!need_geometry)
            continue;

        // tx_axis cotangent (value_0 = tx_axis * (1 + 0j)), then the shared
        // straight offset: direction = safe_normalize(target - source, e_z)
        // and L = safe_length(target - source).
        const field::float3a g_tx_axis = field::make_f3(g_chain.x.re, g_chain.y.re, g_chain.z.re);
        field::float3a g_pol_dump = field::f3_zero();
        ad::adj_transverse_project(chain.direction, load3f(tx_polarization, index), g_tx_axis, g_direction, g_pol_dump);
        const field::float3a offset = field::f3_sub(load3f(target, index), load3f(source, index));
        field::float3a g_offset = field::f3_zero();
        field::float3a g_ez_dump = field::f3_zero();
        field::adj_safe_normalize(offset, field::make_f3(0.0f, 0.0f, 1.0f), g_direction, g_offset, g_ez_dump);
        ad::adj_safe_length(offset, g_total_length, g_offset);
        const int64_t base = index * 3;
        grad_target[base] = g_offset.x;
        grad_target[base + 1] = g_offset.y;
        grad_target[base + 2] = g_offset.z;
        grad_source[base] = -g_offset.x;
        grad_source[base + 1] = -g_offset.y;
        grad_source[base + 2] = -g_offset.z;
    }
}

__global__ void transmission_sequence_jvp_kernel(
    int64_t count, int64_t depth, const bool* path_valid, const float* source, const float* target,
    const float* interaction_normals, const int* interaction_material_id, const bool* interaction_valid,
    const float* tx_power, const float* tx_polarization, const float* rx_polarization, const int* layer_offset,
    const int* layer_count, const float* layer_thickness_m, const float* layer_eps_r, const float* layer_sigma_e,
    const float* layer_mu_r, int64_t material_count, float frequency_hz, const float* tangent_layer_thickness,
    const float* tangent_layer_eps_r, const float* tangent_layer_sigma_e, float tangent_frequency,
    const float* tangent_source, const float* tangent_target, const float* tangent_normals,
    c10::complex<float>* t_field_vector, c10::complex<float>* t_coefficient, c10::complex<float>* t_path_field,
    float* t_path_gain, float* t_path_length, float* t_delay) {
    const em::LayerView layers_base{
        layer_offset, layer_count, layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r, 0,
    };
    const TangentSeed tangent_seed{tangent_layer_thickness, tangent_layer_eps_r, tangent_layer_sigma_e};
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t base = index * 3;
        if (!path_valid[index]) {
            t_field_vector[base] = c10::complex<float>(0.0F, 0.0F);
            t_field_vector[base + 1] = c10::complex<float>(0.0F, 0.0F);
            t_field_vector[base + 2] = c10::complex<float>(0.0F, 0.0F);
            t_coefficient[index] = c10::complex<float>(0.0F, 0.0F);
            t_path_field[index] = c10::complex<float>(0.0F, 0.0F);
            t_path_gain[index] = 0.0F;
            t_path_length[index] = 0.0F;
            t_delay[index] = 0.0F;
            continue;
        }
        TransmissionChain chain;
        transmission_chain_eval(index, depth, source, target, interaction_normals, interaction_material_id,
                                interaction_valid, tx_power, tx_polarization, rx_polarization, layers_base,
                                material_count, frequency_hz, chain);
        if (!chain.path_valid) {
            const int64_t base = index * 3;
            const c10::complex<float> zero(0.0f, 0.0f);
            t_field_vector[base] = zero;
            t_field_vector[base + 1] = zero;
            t_field_vector[base + 2] = zero;
            t_coefficient[index] = zero;
            t_path_field[index] = zero;
            t_path_gain[index] = 0.0f;
            t_path_length[index] = 0.0f;
            t_delay[index] = 0.0f;
            continue;
        }

        // Straight-ray geometry duals shared by every wall: the offset feeds
        // the direction (tx/rx bases and wall frames) and the raw length
        // (amplitude spread, carrier start, path_length_m / delay_s).
        const ad::DualF3 e_z = ad::df3_const(field::make_f3(0.0f, 0.0f, 1.0f));
        const ad::DualF3 offset =
            ad::df3_sub(load_dual3f(target, tangent_target, index), load_dual3f(source, tangent_source, index));
        const ad::DualF total_length = ad::dual_safe_length(offset);
        const ad::DualF3 direction = ad::dual_safe_normalize(offset, e_z);
        const ad::DualF3 tx_axis =
            ad::dual_transverse_project(direction, ad::df3_const(load3f(tx_polarization, index)));
        const ad::DualF3 rx_axis =
            ad::dual_transverse_project(direction, ad::df3_const(load3f(rx_polarization, index)));
        field::Complex3 d_value = field::cplx_scale_real(tx_axis.d, field::cplx(1.0f, 0.0f));
        float d_carrier = total_length.d;
        for (int64_t wall = 0; wall < depth; ++wall) {
            const int material = chain.wall_material[wall];
            if (material < 0)
                continue;
            const ad::DualWallFrame frame =
                ad::dual_wall_frame(direction,
                                    load_dual_sequence3f(interaction_normals, tangent_normals, index, wall, depth));
            em::LayerView layers = layers_base;
            layers.material = material;
            const ad::DualStackRT te = ad::stack_rt_dual(frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                                                         tangent_frequency, em::kPolTE, tangent_seed);
            const ad::DualStackRT tm = ad::stack_rt_dual(frame.cos_theta.v, layers, frequency_hz, frame.cos_theta.d,
                                                         tangent_frequency, em::kPolTM, tangent_seed);
            const field::Complex e_s = chain.e_s[wall];
            const field::Complex e_p = chain.e_p[wall];
            const field::Complex3 value_in = chain.value_in[wall];
            const field::Complex d_e_s = field::cplx_add(transport::complex3_dot_real(d_value, frame.s_axis.v),
                                                         transport::complex3_dot_real(value_in, frame.s_axis.d));
            const field::Complex d_e_p = field::cplx_add(transport::complex3_dot_real(d_value, frame.p_axis.v),
                                                         transport::complex3_dot_real(value_in, frame.p_axis.d));
            const field::Complex w_te = field::cplx_mul(te.t.v, e_s);
            const field::Complex w_tm = field::cplx_mul(tm.t.v, e_p);
            const field::Complex d_w_te = field::cplx_add(field::cplx_mul(te.t.d, e_s), field::cplx_mul(te.t.v, d_e_s));
            const field::Complex d_w_tm = field::cplx_add(field::cplx_mul(tm.t.d, e_p), field::cplx_mul(tm.t.v, d_e_p));
            d_value = field::c3_add(field::c3_add(field::cplx_scale_real(frame.s_axis.d, w_te),
                                                  field::cplx_scale_real(frame.s_axis.v, d_w_te)),
                                    field::c3_add(field::cplx_scale_real(frame.p_axis.d, w_tm),
                                                  field::cplx_scale_real(frame.p_axis.v, d_w_tm)));
            // Carrier chord: d(wall_thickness * cos_theta) with the clamped
            // per-layer thickness gates of the primal accumulation.
            float d_wall_thickness = 0.0f;
            if (tangent_layer_thickness != nullptr) {
                const int first = layer_offset[material];
                const int layers_in_wall = layer_count[material];
                for (int layer = 0; layer < layers_in_wall; ++layer) {
                    const int slot = first + layer;
                    if (layer_thickness_m[slot] >= 0.0f)
                        d_wall_thickness += tangent_layer_thickness[slot];
                }
            }
            d_carrier -= d_wall_thickness * frame.cos_theta.v + chain.wall_thickness[wall] * frame.cos_theta.d;
        }
        const field::Complex d_propagation =
            field::cplx_add(field::cplx_add(field::cplx_mul_real(chain.propagation_dfreq, tangent_frequency),
                                            field::cplx_mul_real(chain.propagation_dcarrier, d_carrier)),
                            field::cplx_mul_real(chain.propagation_dtotal, total_length.d));
        const field::Complex3 value_final = field::c3_scale(chain.value_chain, chain.propagation);
        const field::Complex3 d_final = field::c3_add(field::c3_scale(d_value, chain.propagation),
                                                      field::c3_scale(chain.value_chain, d_propagation));
        write_output_tangents(index, value_final, d_final, chain.rx_axis, rx_axis.d, chain.amplitude_scale,
                              total_length.d, t_field_vector, t_coefficient, t_path_field, t_path_gain, t_path_length,
                              t_delay);
    }
}

std::pair<int64_t, int64_t> check_transmission_ad_primal(const rayd::torch::TransmissionSequenceRequest& request) {
    const auto& source = request.source;
    const auto& path_valid = request.path_valid;
    const auto& target = request.target;
    const auto& interaction_positions = request.interaction_positions;
    const auto& interaction_normals = request.interaction_normals;
    const auto& interaction_material_id = request.interaction_material_id;
    const auto& interaction_valid = request.interaction_valid;
    const auto& tx_power = request.tx_power;
    const auto& tx_polarization = request.tx_polarization;
    const auto& rx_polarization = request.rx_polarization;
    const auto& layer_offset = request.layer_offset;
    const auto& layer_count = request.layer_count;
    const auto& layer_thickness_m = request.layer_thickness_m;
    const auto& layer_eps_r = request.layer_eps_r;
    const auto& layer_sigma_e = request.layer_sigma_e;
    const auto& layer_mu_r = request.layer_mu_r;

    check_vec3_table(source, "source");
    check_flat_tensor(path_valid, "path_valid", at::kBool);
    check_vec3_table(target, "target");
    check_tensor(interaction_positions, "interaction_positions", at::kFloat, 3);
    check_tensor(interaction_normals, "interaction_normals", at::kFloat, 3);
    check_tensor(interaction_material_id, "interaction_material_id", at::kInt, 2);
    check_tensor(interaction_valid, "interaction_valid", at::kBool, 2);
    check_flat_tensor(tx_power, "tx_power", at::kFloat);
    check_vec3_table(tx_polarization, "tx_polarization");
    check_vec3_table(rx_polarization, "rx_polarization");
    check_flat_tensor(layer_offset, "layer_offset", at::kInt);
    check_flat_tensor(layer_count, "layer_count", at::kInt);
    check_flat_tensor(layer_thickness_m, "layer_thickness_m", at::kFloat);
    check_flat_tensor(layer_eps_r, "layer_eps_r", at::kFloat);
    check_flat_tensor(layer_sigma_e, "layer_sigma_e", at::kFloat);
    check_flat_tensor(layer_mu_r, "layer_mu_r", at::kFloat);

    const int64_t count = source.size(0);
    const int64_t depth = interaction_positions.size(1);
    TORCH_CHECK(depth > 0 && depth <= kMaxAdDepth && interaction_positions.size(2) == 3,
                "interaction_positions must have shape (N, D, 3) with 0 < D <= ", kMaxAdDepth);
    TORCH_CHECK(interaction_positions.size(0) == count && interaction_normals.sizes() == interaction_positions.sizes(),
                "interaction tensors must match source rows");
    TORCH_CHECK(interaction_material_id.size(0) == count && interaction_material_id.size(1) == depth &&
                    interaction_valid.size(0) == count && interaction_valid.size(1) == depth,
                "transmission event tensors must have shape (N, D)");
    TORCH_CHECK(path_valid.size(0) == count, "path_valid must match source rows");
    TORCH_CHECK(target.size(0) == count && tx_power.size(0) == count && tx_polarization.size(0) == count &&
                    rx_polarization.size(0) == count,
                "transmission endpoint tensors must match source rows");

    const int64_t material_count = layer_offset.size(0);
    const int64_t layer_total = layer_thickness_m.size(0);
    TORCH_CHECK(layer_count.size(0) == material_count, "layer_count must match layer_offset rows");
    for (const auto& tensor : {layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.size(0) == layer_total, "layer parameter tensors must match layer_thickness_m rows");
    for (const auto& tensor : {path_valid, target, interaction_positions, interaction_normals, interaction_material_id,
                               interaction_valid, tx_power, tx_polarization, rx_polarization, layer_offset, layer_count,
                               layer_thickness_m, layer_eps_r, layer_sigma_e, layer_mu_r})
        TORCH_CHECK(tensor.get_device() == source.get_device(), "transmission tensors must share one CUDA device");
    TORCH_CHECK(request.frequency_hz > 0.0, "frequency_hz must be positive");
    return {depth, material_count};
}

at::Tensor zero_filled(at::IntArrayRef sizes, const at::TensorOptions& options) {
    auto tensor = at::empty(sizes, options);
    if (tensor.numel() > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
        C10_CUDA_CHECK(
            cudaMemsetAsync(tensor.data_ptr(), 0, static_cast<size_t>(tensor.numel()) * tensor.element_size(), stream));
    }
    return tensor;
}

const at::Tensor* optional_tensor(const std::optional<at::Tensor>& value, at::Tensor& storage, const char* name,
                                  at::ScalarType dtype, at::IntArrayRef sizes, const at::Tensor& reference) {
    if (!value.has_value())
        return nullptr;
    storage = value->contiguous();
    TORCH_CHECK(storage.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(storage.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(storage.sizes() == sizes, name, " has the wrong shape");
    TORCH_CHECK(storage.get_device() == reference.get_device(), name, " must share the primal device");
    return &storage;
}

template <typename T> const T* tensor_ptr(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : tensor->data_ptr<T>();
}

} // namespace

rayd::torch::TransmissionSequenceBackwardResult

rayd::torch::field_transmission_sequence_backward(const rayd::torch::TransmissionSequenceBackwardRequest& request) {
    const auto [depth, material_count] = check_transmission_ad_primal(request.primal);
    const auto& primal = request.primal;
    const c10::cuda::CUDAGuard guard(static_cast<int>(primal.source.get_device()));
    const int64_t count = primal.source.size(0);
    const int64_t layer_total = primal.layer_thickness_m.size(0);

    at::Tensor gfv_storage;
    at::Tensor gc_storage;
    at::Tensor gpf_storage;
    at::Tensor gpg_storage;
    at::Tensor gpl_storage;
    at::Tensor gd_storage;
    const at::Tensor* gfv = optional_tensor(request.grad_field_vector, gfv_storage, "grad_field_vector",
                                            at::kComplexFloat, {count, 3}, primal.source);
    const at::Tensor* gc = optional_tensor(request.grad_coefficient, gc_storage, "grad_coefficient", at::kComplexFloat,
                                           {count}, primal.source);
    const at::Tensor* gpf = optional_tensor(request.grad_path_field, gpf_storage, "grad_path_field", at::kComplexFloat,
                                            {count}, primal.source);
    const at::Tensor* gpg =
        optional_tensor(request.grad_path_gain, gpg_storage, "grad_path_gain", at::kFloat, {count}, primal.source);
    const at::Tensor* gpl = optional_tensor(request.grad_path_length_m, gpl_storage, "grad_path_length_m", at::kFloat,
                                            {count}, primal.source);
    const at::Tensor* gd =
        optional_tensor(request.grad_delay_s, gd_storage, "grad_delay_s", at::kFloat, {count}, primal.source);

    auto layer_grad = [&](bool needed) {
        return needed ? zero_filled({layer_total}, primal.source.options()) : at::Tensor();
    };
    at::Tensor grad_thickness = layer_grad(request.need_grad_layer_thickness_m);
    at::Tensor grad_eps = layer_grad(request.need_grad_layer_eps_r);
    at::Tensor grad_sigma = layer_grad(request.need_grad_layer_sigma_e);
    at::Tensor grad_frequency = request.need_grad_frequency ? zero_filled({1}, primal.source.options()) : at::Tensor();
    at::Tensor grad_source;
    at::Tensor grad_target;
    at::Tensor grad_normals;
    if (request.need_grad_geometry) {
        grad_source = zero_filled({count, 3}, primal.source.options());
        grad_target = zero_filled({count, 3}, primal.source.options());
        grad_normals = zero_filled({count, depth, 3}, primal.source.options());
    }
    const bool any_grad_in =
        gfv != nullptr || gc != nullptr || gpf != nullptr || gpg != nullptr || gpl != nullptr || gd != nullptr;
    const bool any_grad_out = request.need_grad_layer_thickness_m || request.need_grad_layer_eps_r ||
                              request.need_grad_layer_sigma_e || request.need_grad_frequency ||
                              request.need_grad_geometry;
    if (count > 0 && any_grad_in && any_grad_out) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(primal.source.get_device()).stream();
        transmission_sequence_backward_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, primal.path_valid.data_ptr<bool>(), primal.source.data_ptr<float>(),
            primal.target.data_ptr<float>(), primal.interaction_normals.data_ptr<float>(),
            primal.interaction_material_id.data_ptr<int>(), primal.interaction_valid.data_ptr<bool>(),
            primal.tx_power.data_ptr<float>(), primal.tx_polarization.data_ptr<float>(),
            primal.rx_polarization.data_ptr<float>(), primal.layer_offset.data_ptr<int>(),
            primal.layer_count.data_ptr<int>(), primal.layer_thickness_m.data_ptr<float>(),
            primal.layer_eps_r.data_ptr<float>(), primal.layer_sigma_e.data_ptr<float>(),
            primal.layer_mu_r.data_ptr<float>(), material_count, static_cast<float>(primal.frequency_hz),
            gfv ? gfv->data_ptr<c10::complex<float>>() : nullptr, gc ? gc->data_ptr<c10::complex<float>>() : nullptr,
            gpf ? gpf->data_ptr<c10::complex<float>>() : nullptr, tensor_ptr<float>(gpg), tensor_ptr<float>(gpl),
            tensor_ptr<float>(gd), request.need_grad_layer_thickness_m ? grad_thickness.data_ptr<float>() : nullptr,
            request.need_grad_layer_eps_r ? grad_eps.data_ptr<float>() : nullptr,
            request.need_grad_layer_sigma_e ? grad_sigma.data_ptr<float>() : nullptr,
            request.need_grad_frequency ? grad_frequency.data_ptr<float>() : nullptr,
            request.need_grad_geometry ? grad_source.data_ptr<float>() : nullptr,
            request.need_grad_geometry ? grad_target.data_ptr<float>() : nullptr,
            request.need_grad_geometry ? grad_normals.data_ptr<float>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return {request.need_grad_layer_thickness_m ? std::optional<at::Tensor>(grad_thickness) : std::nullopt,
            request.need_grad_layer_eps_r ? std::optional<at::Tensor>(grad_eps) : std::nullopt,
            request.need_grad_layer_sigma_e ? std::optional<at::Tensor>(grad_sigma) : std::nullopt,
            request.need_grad_frequency ? std::optional<at::Tensor>(grad_frequency) : std::nullopt,
            request.need_grad_geometry ? std::optional<at::Tensor>(grad_source) : std::nullopt,
            request.need_grad_geometry ? std::optional<at::Tensor>(grad_target) : std::nullopt,
            std::nullopt,
            request.need_grad_geometry ? std::optional<at::Tensor>(grad_normals) : std::nullopt};
}

rayd::torch::TransmissionSequenceJvpResult rayd::torch::field_transmission_sequence_jvp(
    const rayd::torch::TransmissionSequenceJvpRequest& request) {
    const auto [depth, material_count] = check_transmission_ad_primal(request.primal);
    const auto& primal = request.primal;
    const c10::cuda::CUDAGuard guard(static_cast<int>(primal.source.get_device()));
    const int64_t count = primal.source.size(0);
    const int64_t layer_total = primal.layer_thickness_m.size(0);
    at::Tensor tt_storage;
    at::Tensor te_storage;
    at::Tensor ts_storage;
    at::Tensor tsrc_storage;
    at::Tensor ttgt_storage;
    at::Tensor tpos_storage;
    at::Tensor tnrm_storage;
    const at::Tensor* t_thickness =
        optional_tensor(request.tangent_layer_thickness_m, tt_storage, "tangent_layer_thickness_m", at::kFloat,
                        {layer_total}, primal.source);
    const at::Tensor* t_eps = optional_tensor(request.tangent_layer_eps_r, te_storage, "tangent_layer_eps_r",
                                              at::kFloat, {layer_total}, primal.source);
    const at::Tensor* t_sigma = optional_tensor(request.tangent_layer_sigma_e, ts_storage, "tangent_layer_sigma_e",
                                                at::kFloat, {layer_total}, primal.source);
    const at::Tensor* t_source =
        optional_tensor(request.tangent_source, tsrc_storage, "tangent_source", at::kFloat, {count, 3}, primal.source);
    const at::Tensor* t_target =
        optional_tensor(request.tangent_target, ttgt_storage, "tangent_target", at::kFloat, {count, 3}, primal.source);
    (void)optional_tensor(request.tangent_interaction_positions, tpos_storage, "tangent_interaction_positions",
                          at::kFloat, {count, depth, 3}, primal.source);
    const at::Tensor* t_normals =
        optional_tensor(request.tangent_interaction_normals, tnrm_storage, "tangent_interaction_normals", at::kFloat,
                        {count, depth, 3}, primal.source);

    auto complex_options = primal.source.options().dtype(at::kComplexFloat);
    auto t_field_vector = at::empty({count, 3}, complex_options);
    auto t_coefficient = at::empty({count}, complex_options);
    auto t_path_field = at::empty({count}, complex_options);
    auto t_path_gain = at::empty({count}, primal.source.options());
    auto t_path_length = at::empty({count}, primal.source.options());
    auto t_delay = at::empty({count}, primal.source.options());
    if (count > 0) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(primal.source.get_device()).stream();
        transmission_sequence_jvp_kernel<<<launch_blocks(count), kBlockSize, 0, stream>>>(
            count, depth, primal.path_valid.data_ptr<bool>(), primal.source.data_ptr<float>(),
            primal.target.data_ptr<float>(), primal.interaction_normals.data_ptr<float>(),
            primal.interaction_material_id.data_ptr<int>(), primal.interaction_valid.data_ptr<bool>(),
            primal.tx_power.data_ptr<float>(), primal.tx_polarization.data_ptr<float>(),
            primal.rx_polarization.data_ptr<float>(), primal.layer_offset.data_ptr<int>(),
            primal.layer_count.data_ptr<int>(), primal.layer_thickness_m.data_ptr<float>(),
            primal.layer_eps_r.data_ptr<float>(), primal.layer_sigma_e.data_ptr<float>(),
            primal.layer_mu_r.data_ptr<float>(), material_count, static_cast<float>(primal.frequency_hz),
            tensor_ptr<float>(t_thickness), tensor_ptr<float>(t_eps), tensor_ptr<float>(t_sigma),
            static_cast<float>(request.tangent_frequency), tensor_ptr<float>(t_source), tensor_ptr<float>(t_target),
            tensor_ptr<float>(t_normals), t_field_vector.data_ptr<c10::complex<float>>(),
            t_coefficient.data_ptr<c10::complex<float>>(), t_path_field.data_ptr<c10::complex<float>>(),
            t_path_gain.data_ptr<float>(), t_path_length.data_ptr<float>(), t_delay.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {t_field_vector, t_coefficient, t_path_field, t_path_gain, t_path_length, t_delay};
}
