cmake_minimum_required(VERSION 3.26)
project(rayd_torch LANGUAGES CXX CUDA)
set(RAYD_SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for RayD Torch native targets." FORCE)
endif()

set(RAYD_TORCH_DEFAULT_CUDA_ARCHITECTURES "70-real;75-real;80-real;86-real;89-real;90-real;100-real;101-real;120-real;120-virtual")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(
        CMAKE_CUDA_ARCHITECTURES
        "${RAYD_TORCH_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures for RayD Torch native kernels."
        FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(RAYD_TORCH_BUILD_NATIVE "Build the RayD Torch extension." ON)
option(RAYD_TORCH_BUILD_PYTHON_MODULE "Build the rayd.torch Python extension module." ON)
option(RAYD_TORCH_OPTIX_FAST_MATH "Compile embedded OptiX PTX with --use_fast_math." ON)

set(RAYD_TORCH_OPTIX_NVCC_FLAGS --std=c++17 --gpu-architecture=compute_75)
if(RAYD_TORCH_OPTIX_FAST_MATH)
    list(APPEND RAYD_TORCH_OPTIX_NVCC_FLAGS --use_fast_math)
endif()

if(RAYD_TORCH_BUILD_NATIVE)
    find_package(CUDAToolkit 11.0 REQUIRED)
    find_path(
        OPTIX_INCLUDE_DIR
        optix.h
        HINTS
            ENV OPTIX_INCLUDE_DIR
            ENV OPTIX_PATH
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 9.1.0"
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.1.0"
            "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.0.0"
            "/usr/local/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64"
            "/opt/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64"
        PATH_SUFFIXES include
    )
    if(NOT OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR "Could not locate OptiX SDK include directory. Set OPTIX_INCLUDE_DIR or OPTIX_PATH.")
    endif()

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
        message(FATAL_ERROR "Could not locate PyTorch CMake prefix. Install torch in the build environment.")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    if(WIN32 AND NOT EXISTS "${TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
        get_filename_component(PYTHON_ENV_ROOT "${Python_EXECUTABLE}" DIRECTORY)
        list(PREPEND CMAKE_PREFIX_PATH
            "${PYTHON_ENV_ROOT}/Library"
            "${PYTHON_ENV_ROOT}/Lib/site-packages")
        set(CONDA_TORCH_DIR "${PYTHON_ENV_ROOT}/Library/share/cmake/Torch")
        if(EXISTS "${CONDA_TORCH_DIR}/TorchConfig.cmake")
            set(Torch_DIR "${CONDA_TORCH_DIR}")
        endif()
    endif()
    # PyTorch's consumed Caffe2 config (Caffe2/public/cuda.cmake) sets the
    # normal variable CMAKE_CUDA_ARCHITECTURES to OFF, which would strip all
    # -gencode flags from every target created below and silently fall back to
    # nvcc's default architecture. Save the requested value and restore it.
    set(RAYD_TORCH_REQUESTED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    if(NOT DEFINED TORCH_CUDA_ARCH_LIST OR TORCH_CUDA_ARCH_LIST STREQUAL "")
        set(TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6;8.9;9.0;10.0;10.1;12.0+PTX")
    endif()
    find_package(Torch REQUIRED)
    if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${RAYD_TORCH_REQUESTED_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES "${RAYD_TORCH_REQUESTED_CUDA_ARCHITECTURES}")
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
        message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES resolved to '${CMAKE_CUDA_ARCHITECTURES}'; RayD Torch kernels would build without -gencode flags.")
    endif()
    message(STATUS "RayD Torch CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "RayD Torch/Caffe2 CUDA architectures: ${TORCH_CUDA_ARCH_LIST}")

    # Stable ABI shims are exported by torch_cpu and torch_cuda. Link those
    # provider libraries directly: ${TORCH_LIBRARIES} also pulls in the
    # unstable c10 ABI and, on Windows, forces an at::cuda::warp_size import.
    find_library(
        RAYD_TORCH_STABLE_CPU_LIBRARY
        NAMES torch_cpu
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
        REQUIRED)
    find_library(
        RAYD_TORCH_STABLE_CUDA_LIBRARY
        NAMES torch_cuda
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
        REQUIRED)

    # This library is intentionally not a Python extension. It uses only the
    # LibTorch Stable ABI and is loaded from Python through torch.ops.load_library().
    add_library(rayd_torch_stable_ops SHARED src/stable/camera.cu)
    set_target_properties(
        rayd_torch_stable_ops
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME "_stable_ops"
            POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(
        rayd_torch_stable_ops
        PRIVATE
            TORCH_TARGET_VERSION=0x020a000000000000
            USE_CUDA)
    target_include_directories(
        rayd_torch_stable_ops
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            "${TORCH_INSTALL_PREFIX}/include"
            ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(
        rayd_torch_stable_ops
        PRIVATE
            "${RAYD_TORCH_STABLE_CPU_LIBRARY}"
            "${RAYD_TORCH_STABLE_CUDA_LIBRARY}"
            CUDA::cudart
            CUDA::cuda_driver)
    target_compile_features(rayd_torch_stable_ops PRIVATE cxx_std_17)
    install(
        TARGETS rayd_torch_stable_ops
        LIBRARY DESTINATION rayd/torch
        RUNTIME DESTINATION rayd/torch)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)"
        RESULT_VARIABLE TORCH_PYTHON_PACKAGE_RESULT
        OUTPUT_VARIABLE TORCH_PYTHON_PACKAGE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT TORCH_PYTHON_PACKAGE_RESULT EQUAL 0)
        message(FATAL_ERROR "Could not locate the PyTorch Python package directory.")
    endif()
    find_library(
        TORCH_PYTHON_LIBRARY
        torch_python
        PATHS
            "${TORCH_INSTALL_PREFIX}/lib"
            "${TORCH_PYTHON_PACKAGE_DIR}/lib"
        REQUIRED)

    function(pybind11_add_module target)
        Python_add_library(${target} MODULE WITH_SOABI ${ARGN})
        target_include_directories(
            ${target}
            PRIVATE
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
        )
    endfunction()

    set(RAYD_TORCH_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_intersect.ptx")
    set(RAYD_TORCH_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/optix_intersect_ptx.h")
    set(RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_point_ray.ptx")
    set(RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/edge_optix_point_ray_ptx.h")
    set(RAYD_TORCH_EDGE_OPTIX_TOPK_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_topk.ptx")
    set(RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/edge_optix_topk_ptx.h")
    set(RAYD_TORCH_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_optix.ptx")
    set(RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/reflection_trace_optix_ptx.h")
    set(RAYD_TORCH_SEGMENT_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_visibility_optix.ptx")
    set(RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/segment_visibility_optix_ptx.h")
    set(RAYD_TORCH_REFLECTION_EPC_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_optix.ptx")
    set(RAYD_TORCH_REFLECTION_EPC_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/reflection_epc_optix_ptx.h")
    set(RAYD_TORCH_REFLECTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_accumulation_optix.ptx")
    set(RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/reflection_accumulation_optix_ptx.h")
    set(RAYD_TORCH_DIFFRACTION_PATHS_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_paths_optix.ptx")
    set(RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/diffraction_paths_optix_ptx.h")
    set(RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_optix.ptx")
    set(RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/torch/diffraction_accumulation_optix_ptx.h")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/scene/optix_intersect.cu"
            -o "${RAYD_TORCH_OPTIX_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/scene/optix_intersect.cu
            include/rayd/torch/scene/optix_intersect_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_OPTIX_PTX}"
            "${RAYD_TORCH_OPTIX_PTX_HEADER}"
            rayd_torch_optix_intersect_ptx
        DEPENDS
            "${RAYD_TORCH_OPTIX_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_optix_ptx DEPENDS "${RAYD_TORCH_OPTIX_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            -DRAYD_TORCH_EDGE_POINT_RAY_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            rayd_torch_edge_optix_point_ray_ptx
        DEPENDS
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/edge/edge_optix.cu"
            -o "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            -DRAYD_TORCH_EDGE_TOPK_ONLY=1
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/edge/edge_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/edge/optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
            rayd_torch_edge_optix_topk_ptx
        DEPENDS
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(
        rayd_torch_edge_optix_ptx
        DEPENDS
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/trace_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/trace_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/reflection/trace_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}"
            rayd_torch_reflection_trace_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_trace_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/visibility_optix.cu"
            -o "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/visibility_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/reflection/visibility_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
            rayd_torch_segment_visibility_optix_ptx
        DEPENDS
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_segment_visibility_optix_ptx DEPENDS "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_EPC_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/epc_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/epc_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/reflection/epc_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}"
            rayd_torch_reflection_epc_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_epc_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/reflection/accum_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/reflection/accum_optix.cu
            include/rayd/torch/common/complex.cuh
            include/rayd/torch/common/math.cuh
            include/rayd/torch/reflection/accum_params.h
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
            rayd_torch_reflection_accumulation_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_accumulation_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/paths_optix.cu"
            -o "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${RAYD_SHARED_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/paths_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/diffraction/common.h
            include/rayd/torch/diffraction/paths_params.h
            "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
            "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}"
            rayd_torch_diffraction_paths_optix_ptx
        DEPENDS
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_diffraction_paths_optix_ptx DEPENDS "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/diffraction/accum_optix.cu"
            -o "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${RAYD_SHARED_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            src/torch_ext/diffraction/accum_optix.cu
            include/rayd/torch/common/math.cuh
            include/rayd/torch/diffraction/common.h
            include/rayd/torch/diffraction/accum_params.h
            "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
            "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
            rayd_torch_diffraction_accumulation_optix_ptx
        DEPENDS
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(rayd_torch_diffraction_accumulation_optix_ptx DEPENDS "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}")

    set(
        RAYD_TORCH_NATIVE_CORE_SOURCES
        src/torch_ext/tensor_check.cpp
        src/torch_ext/scene/scene_cache.cpp
        src/torch_ext/common/optix_pipeline.cpp
        src/torch_ext/common/ops_stats.cpp
        src/torch_ext/common/ops_camera.cpp
        src/torch_ext/reflection/pipeline.cpp
        src/torch_ext/diffraction/pipeline.cpp
        src/torch_ext/scene/ops_scene.cpp
        src/torch_ext/scene/optix_context.cpp
        src/torch_ext/scene/ops_intersect.cpp
        src/torch_ext/edge/ops_edge.cpp
        src/torch_ext/reflection/ops.cpp
        src/torch_ext/diffraction/ops.cpp
        src/torch_ext/scene/cache_kernels.cu
        src/torch_ext/scene/geometry_forward.cu
        src/torch_ext/scene/geometry_backward.cu
        src/torch_ext/edge/bvh.cu
        src/torch_ext/edge/edge_forward.cu
        src/torch_ext/edge/edge_backward.cu
        src/torch_ext/reflection/visibility_backward.cu
        src/torch_ext/reflection/backward.cu
        src/torch_ext/reflection/dedup.cu
        src/torch_ext/reflection/epc_field.cu
        src/torch_ext/reflection/accum_reduce.cu
        src/torch_ext/diffraction/accum_reduce.cu
        src/torch_ext/diffraction/accum_ad.cu
        src/torch_ext/diffraction/paths_init.cu
        src/torch_ext/common/stats.cu
        src/torch_ext/common/camera.cu
    )

    add_library(rayd_torch_native_core STATIC ${RAYD_TORCH_NATIVE_CORE_SOURCES})
    set_target_properties(rayd_torch_native_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
        rayd_torch_native_core
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${RAYD_SHARED_INCLUDE_DIR}
        PRIVATE
            ${Python_INCLUDE_DIRS}
            "${TORCH_INSTALL_PREFIX}/include"
            "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    add_dependencies(
        rayd_torch_native_core
        rayd_torch_optix_ptx
        rayd_torch_edge_optix_ptx
        rayd_torch_reflection_trace_optix_ptx
        rayd_torch_segment_visibility_optix_ptx
        rayd_torch_reflection_epc_optix_ptx
        rayd_torch_reflection_accumulation_optix_ptx
        rayd_torch_diffraction_paths_optix_ptx
        rayd_torch_diffraction_accumulation_optix_ptx)
    target_link_libraries(
        rayd_torch_native_core
        PUBLIC
            "${TORCH_LIBRARIES}"
            "${TORCH_PYTHON_LIBRARY}"
            CUDA::cudart
            CUDA::cuda_driver)
    target_compile_features(rayd_torch_native_core PUBLIC cxx_std_17)

    if(RAYD_TORCH_BUILD_PYTHON_MODULE)
        pybind11_add_module(
            _C
            src/torch_ext/module.cpp
            src/torch_ext/library.cpp
        )
        target_include_directories(
            _C
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${RAYD_SHARED_INCLUDE_DIR}
                ${CMAKE_CURRENT_BINARY_DIR}/generated
                ${CUDAToolkit_INCLUDE_DIRS}
                ${OPTIX_INCLUDE_DIR}
        )
        target_link_libraries(_C PRIVATE rayd_torch_native_core)
        target_compile_features(_C PRIVATE cxx_std_17)
        install(TARGETS _C LIBRARY DESTINATION rayd/torch RUNTIME DESTINATION rayd/torch)
    endif()
endif()
