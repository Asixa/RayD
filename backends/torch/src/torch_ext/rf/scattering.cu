


#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <rayd/torch/rf/scattering.h>

#include "scattering_tensor_checks.h"
#include <rayd/shared/rf/scattering_table.cuh>

namespace {

constexpr int kBlockSize = 256;
constexpr float kTwoPi = rayd::shared::rf::scattering_tables::kTwoPi;

using rayd::shared::rf::scattering_tables::interp4;
using rayd::shared::rf::scattering_tables::linear_axis;
using rayd::shared::rf::scattering_tables::nearest_axis;
using rayd::shared::rf::scattering_tables::positive_phi;

__device__ __forceinline__ float table_pdf(
    const float* __restrict__ density, int nti, int npi, int nto, int npo,
    const float* wi, const float* wo) {
    if (wi[2] <= 0.0f || wo[2] <= 0.0f) return 0.0f;
    const float phi_i = positive_phi(wi[1], wi[0]);
    float phi_o = positive_phi(wo[1], wo[0]);
    if (npi == 1) { phi_o -= phi_i; if (phi_o < 0.0f) phi_o += kTwoPi; }
    const int ti = nearest_axis(wi[2], nti, 1.0f, false);
    const int pi = npi == 1 ? 0 : nearest_axis(phi_i, npi, kTwoPi, true);
    const int to = nearest_axis(wo[2], nto, 1.0f, false);
    const int po = nearest_axis(phi_o, npo, kTwoPi, true);
    return density[((static_cast<int64_t>(ti) * npi + pi) * nto + to) * npo + po];
}
__global__ void scattering_eval_kernel(
    int64_t count, const bool* __restrict__ valid,
    const float* __restrict__ wi, const float* __restrict__ wo,
    const float* __restrict__ fte, const float* __restrict__ ftm,
    int nti, int npi, int nto, int npo, float* out_te, float* out_tm) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (!valid[row]) { out_te[row] = 0.0f; out_tm[row] = 0.0f; continue; }
        rayd::shared::rf::scattering_tables::eval_te_tm(
            fte, ftm, nti, npi, nto, npo, wi + 3 * row, wo + 3 * row,
            out_te[row], out_tm[row]);
    }
}

__global__ void scattering_pdf_kernel(
    int64_t count, const bool* valid, const float* wi, const float* wo, const float* density,
    int nti, int npi, int nto, int npo, bool reverse, float* out) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (!valid[row]) { out[row] = 0.0f; continue; }
        const float* a = wi + 3 * row; const float* b = wo + 3 * row;
        out[row] = reverse ? table_pdf(density,nti,npi,nto,npo,b,a)
                           : table_pdf(density,nti,npi,nto,npo,a,b);
    }
}

__global__ void scattering_sample_kernel(
    int64_t count, const bool* valid, const float* wi, const float* uniforms,
    const float* marginal, const float* conditional, const float* density,
    int nti, int npi, int nto, int npo,
    float* wo, float* pdf_fwd, float* pdf_rev) {
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < count; row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if (!valid[row]) {
            float* b = wo + 3 * row;
            b[0] = 0.0f; b[1] = 0.0f; b[2] = 0.0f;
            pdf_fwd[row] = 0.0f; pdf_rev[row] = 0.0f;
            continue;
        }
        const float* a = wi + 3 * row;
        const float phi_i = positive_phi(a[1], a[0]);
        const int ti = nearest_axis(a[2], nti, 1.0f, false);
        const int pi = npi == 1 ? 0 : nearest_axis(phi_i, npi, kTwoPi, true);
        const float u1 = fminf(fmaxf(uniforms[2*row],0.0f),1.0f-1e-7f);
        const float u2 = fminf(fmaxf(uniforms[2*row+1],0.0f),1.0f-1e-7f);
        const int64_t mbase = (static_cast<int64_t>(ti)*npi+pi)*nto;
        int lo=0, hi=nto;
        while (lo<hi) { const int mid=(lo+hi)>>1; if (marginal[mbase+mid]<=u1) lo=mid+1; else hi=mid; }
        const int to=min(lo,nto-1);
        const float mlo = to ? marginal[mbase+to-1] : 0.0f;
        const float mhi = marginal[mbase+to];
        const float mf = mhi > mlo ? (u1-mlo)/(mhi-mlo) : 0.5f;
        const int64_t cbase = ((static_cast<int64_t>(ti)*npi+pi)*nto+to)*npo;
        lo=0; hi=npo;
        while (lo<hi) { const int mid=(lo+hi)>>1; if (conditional[cbase+mid]<=u2) lo=mid+1; else hi=mid; }
        const int po=min(lo,npo-1);
        const float clo = po ? conditional[cbase+po-1] : 0.0f;
        const float chi = conditional[cbase+po];
        const float pf = chi > clo ? (u2-clo)/(chi-clo) : 0.5f;
        const float cos_o = fminf(fmaxf((static_cast<float>(to)+mf)/nto,1e-6f),1.0f);
        float phi_o = (static_cast<float>(po)+pf)*(kTwoPi/npo);
        if (npi == 1) phi_o += phi_i;
        const float sin_o = sqrtf(fmaxf(0.0f,1.0f-cos_o*cos_o));
        float* b=wo+3*row; b[0]=sin_o*cosf(phi_o); b[1]=sin_o*sinf(phi_o); b[2]=cos_o;
        pdf_fwd[row]=density[cbase+po];
        pdf_rev[row]=table_pdf(density,nti,npi,nto,npo,b,a);
    }
}

int blocks(int64_t n) { return static_cast<int>((n+kBlockSize-1)/kBlockSize); }

void check_table(const at::Tensor& t, const char* name) {
    rayd::torch::detail::check_tensor(t,name,at::kFloat,4);
}

} // namespace

rayd::torch::ScatteringTableEvalResult scattering_table_eval_impl(at::Tensor valid, at::Tensor wi, at::Tensor wo, at::Tensor fte, at::Tensor ftm) {
    rayd::torch::detail::check_tensor(valid,"valid",at::kBool,1);
    rayd::torch::detail::check_vec3_table(wi,"wi"); rayd::torch::detail::check_vec3_table(wo,"wo");
    check_table(fte,"f_te"); check_table(ftm,"f_tm");
    TORCH_CHECK(valid.size(0)==wi.size(0),"valid must match wi rows"); TORCH_CHECK(wi.sizes()==wo.sizes(),"wi and wo shapes must match");
    TORCH_CHECK(fte.sizes()==ftm.sizes(),"f_te and f_tm shapes must match");
    TORCH_CHECK(valid.get_device()==wi.get_device() && wi.get_device()==fte.get_device() && wo.get_device()==wi.get_device() && ftm.get_device()==wi.get_device(),"scattering tensors must share device");
    auto te=at::empty({wi.size(0)},wi.options()), tm=at::empty_like(te);
    if (wi.size(0)>0) { auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream(); scattering_eval_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),valid.data_ptr<bool>(),wi.data_ptr<float>(),wo.data_ptr<float>(),fte.data_ptr<float>(),ftm.data_ptr<float>(),fte.size(0),fte.size(1),fte.size(2),fte.size(3),te.data_ptr<float>(),tm.data_ptr<float>()); C10_CUDA_KERNEL_LAUNCH_CHECK(); }
    return {te, tm};
}

at::Tensor scattering_table_pdf_impl(at::Tensor valid, at::Tensor wi, at::Tensor wo, at::Tensor density, bool reverse) {
    rayd::torch::detail::check_tensor(valid,"valid",at::kBool,1);
    rayd::torch::detail::check_vec3_table(wi,"wi"); rayd::torch::detail::check_vec3_table(wo,"wo"); check_table(density,"sample_density");
    TORCH_CHECK(valid.size(0)==wi.size(0),"valid must match wi rows"); TORCH_CHECK(wi.sizes()==wo.sizes(),"wi and wo shapes must match"); TORCH_CHECK(valid.get_device()==wi.get_device() && wi.get_device()==density.get_device(),"scattering tensors must share device");
    auto out=at::empty({wi.size(0)},wi.options()); if(wi.size(0)>0){auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream();scattering_pdf_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),valid.data_ptr<bool>(),wi.data_ptr<float>(),wo.data_ptr<float>(),density.data_ptr<float>(),density.size(0),density.size(1),density.size(2),density.size(3),reverse,out.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();} return out;
}

rayd::torch::ScatteringTableSampleResult scattering_table_sample_impl(at::Tensor valid, at::Tensor wi, at::Tensor uniforms, at::Tensor marginal, at::Tensor conditional, at::Tensor density) {
    rayd::torch::detail::check_tensor(valid,"valid",at::kBool,1);
    rayd::torch::detail::check_vec3_table(wi,"wi"); rayd::torch::detail::check_tensor(uniforms,"uniforms",at::kFloat,2); rayd::torch::detail::check_tensor(marginal,"marginal_cdf",at::kFloat,3); check_table(conditional,"conditional_cdf"); check_table(density,"sample_density");
    TORCH_CHECK(valid.size(0)==wi.size(0),"valid must match wi rows"); TORCH_CHECK(uniforms.size(0)==wi.size(0)&&uniforms.size(1)==2,"uniforms must have shape (N,2)"); TORCH_CHECK(valid.get_device()==wi.get_device()&&marginal.get_device()==wi.get_device()&&conditional.get_device()==wi.get_device()&&density.get_device()==wi.get_device()&&uniforms.get_device()==wi.get_device(),"scattering tensors must share device");
    auto wo=at::empty_like(wi), pf=at::empty({wi.size(0)},wi.options()), pr=at::empty_like(pf); if(wi.size(0)>0){auto s=at::cuda::getCurrentCUDAStream(wi.get_device()).stream();scattering_sample_kernel<<<blocks(wi.size(0)),kBlockSize,0,s>>>(wi.size(0),valid.data_ptr<bool>(),wi.data_ptr<float>(),uniforms.data_ptr<float>(),marginal.data_ptr<float>(),conditional.data_ptr<float>(),density.data_ptr<float>(),density.size(0),density.size(1),density.size(2),density.size(3),wo.data_ptr<float>(),pf.data_ptr<float>(),pr.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();} return {wo, pf, pr};
}

rayd::torch::ScatteringTableEvalResult rayd::torch::scattering_table_eval(
    const ScatteringTableEvalRequest& request) {
    return scattering_table_eval_impl(
        request.valid, request.wi, request.wo, request.f_te, request.f_tm);
}

rayd::torch::ScatteringTablePdfResult rayd::torch::scattering_table_pdf(
    const ScatteringTablePdfRequest& request) {
    return {scattering_table_pdf_impl(
        request.valid, request.wi, request.wo, request.sample_density, request.reverse)};
}

rayd::torch::ScatteringTableSampleResult rayd::torch::scattering_table_sample(
    const ScatteringTableSampleRequest& request) {
    return scattering_table_sample_impl(
        request.valid,
        request.wi,
        request.uniforms,
        request.marginal_cdf,
        request.conditional_cdf,
        request.sample_density);
}
