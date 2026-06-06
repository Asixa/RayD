#include <torch/extension.h>

namespace raydtorch {

py::dict build_info() {
    py::dict info;
    info["backend"] = "raydtorch-native";
    info["uses_drjit"] = false;
    return info;
}

PYBIND11_MODULE(_raydtorch, m) {
    m.doc() = "RayDTorch CUDA/OptiX backend.";
    m.def("build_info", &build_info);
}

} // namespace raydtorch
