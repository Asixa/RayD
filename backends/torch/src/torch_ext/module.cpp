#include <torch/extension.h>

namespace rayd::torch_backend {

py::dict build_info() {
    py::dict info;
    info["backend"] = "torch";
    info["uses_dr_jit"] = false;
    return info;
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "RayD Torch CUDA/OptiX backend.";
    m.def("build_info", &build_info);
}

} // namespace rayd::torch_backend
