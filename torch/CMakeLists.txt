cmake_minimum_required(VERSION 3.26)
project(rayd_torch VERSION 0.7.0 LANGUAGES CXX CUDA)
set(RAYD_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RAYD_INCLUDE_DIR "${RAYD_ROOT_DIR}/include")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(
    RAYD_TORCH_CUDA_GENCODE_FLAGS
    ""
    CACHE STRING
    "Optional explicit NVCC gencode flags used to share CUDA front-end work across compatible real architectures.")
if(DEFINED ENV{RAYD_TORCH_CUDA_GENCODE_FLAGS}
   AND NOT "$ENV{RAYD_TORCH_CUDA_GENCODE_FLAGS}" STREQUAL "")
    set(
        RAYD_TORCH_CUDA_GENCODE_FLAGS
        "$ENV{RAYD_TORCH_CUDA_GENCODE_FLAGS}"
        CACHE STRING
        "Optional explicit NVCC gencode flags used to share CUDA front-end work across compatible real architectures."
        FORCE)
endif()
if(NOT RAYD_TORCH_CUDA_GENCODE_FLAGS STREQUAL "")
    separate_arguments(
        RAYD_TORCH_CUDA_GENCODE_OPTIONS
        NATIVE_COMMAND
        "${RAYD_TORCH_CUDA_GENCODE_FLAGS}")
    set(RAYD_TORCH_CALLER_CUDA_FLAGS "$ENV{CMAKE_CUDA_FLAGS}")
    set(
        CMAKE_CUDA_ARCHITECTURES
        OFF
        CACHE STRING
        "CUDA architectures are supplied by RAYD_TORCH_CUDA_GENCODE_FLAGS."
        FORCE)
endif()

function(rayd_torch_apply_cuda_gencode target)
    foreach(option IN LISTS RAYD_TORCH_CUDA_GENCODE_OPTIONS)
        target_compile_options(
            "${target}"
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:${option}>")
    endforeach()
endfunction()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for RayD Torch native targets." FORCE)
endif()

set(RAYD_TORCH_DEFAULT_CUDA_ARCHITECTURES "native")
if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES}
   AND NOT "$ENV{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(
        CMAKE_CUDA_ARCHITECTURES
        "$ENV{CMAKE_CUDA_ARCHITECTURES}"
        CACHE STRING
        "CUDA architectures for RayD Torch native kernels."
        FORCE)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
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
option(RAYD_TORCH_BUILD_CPP_TESTS "Build direct C++ tests for the typed integration API." OFF)
option(RAYD_TORCH_OPTIX_FAST_MATH "Compile embedded OptiX PTX with --use_fast_math." ON)
option(
    RAYD_TORCH_INSTALL_SOURCE_BUNDLE
    "Install the integrity-described source bundle for same-graph downstream builds."
    ON)
set(
    RAYD_SOURCE_COMMIT
    ""
    CACHE STRING
    "RayD source commit used when a package build has no Git checkout.")
set(
    RAYD_SOURCE_REPOSITORY_URL
    ""
    CACHE STRING
    "RayD repository URL used when a package build has no Git checkout.")
include(CTest)
if(BUILD_TESTING AND RAYD_TORCH_BUILD_CPP_TESTS)
    find_package(Python COMPONENTS Development.Embed REQUIRED)
endif()

if(RAYD_TORCH_INSTALL_SOURCE_BUNDLE)
    set(RAYD_TORCH_SOURCE_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/rayd-source-bundle")
    set(
        RAYD_TORCH_SOURCE_BUNDLE_COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_source_bundle.py"
        --workspace "${RAYD_ROOT_DIR}"
        --output "${RAYD_TORCH_SOURCE_BUNDLE}"
        --distribution-version "${PROJECT_VERSION}")
    if(NOT RAYD_SOURCE_COMMIT STREQUAL "")
        list(APPEND RAYD_TORCH_SOURCE_BUNDLE_COMMAND --commit "${RAYD_SOURCE_COMMIT}")
    endif()
    if(NOT RAYD_SOURCE_REPOSITORY_URL STREQUAL "")
        list(
            APPEND
            RAYD_TORCH_SOURCE_BUNDLE_COMMAND
            --repository-url "${RAYD_SOURCE_REPOSITORY_URL}")
    endif()
    execute_process(
        COMMAND ${RAYD_TORCH_SOURCE_BUNDLE_COMMAND}
        RESULT_VARIABLE RAYD_TORCH_SOURCE_BUNDLE_RESULT
        ERROR_VARIABLE RAYD_TORCH_SOURCE_BUNDLE_ERROR)
    if(NOT RAYD_TORCH_SOURCE_BUNDLE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Could not generate the RayD Torch source bundle: "
            "${RAYD_TORCH_SOURCE_BUNDLE_ERROR}")
    endif()
    install(
        DIRECTORY "${RAYD_TORCH_SOURCE_BUNDLE}/"
        DESTINATION rayd/torch/_source)
endif()

# Each canonical private Python file has exactly one wheel owner.  Keep this
# list exact so rayd-torch can never collect adjacent *_jit.py implementations.
install(
    FILES
        ${RAYD_ROOT_DIR}/python/rayd/_impl/runtime.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/capabilities.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/path_exchange.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/geometry.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/scene.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/multi.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/multipath.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/camera.py
        ${RAYD_ROOT_DIR}/python/rayd/_impl/sdf.py
    DESTINATION rayd/_impl)
# nvcc numeric flags are a contract. contracts/compile_policy.json and
# docs/adr/0035-cuda-compile-flag-policy.md declare the numeric profile of every
# translation unit in this file, and tests/test_compile_flag_policy_contract.py
# re-derives it from here. ADR-0035 profile: fast_math for the ten OptiX PTX
# modules that read the variable below. Divergence D8: this option is the only
# place either backend can be switched between profiles, and Dr.Jit has no
# equivalent. Divergence D2: Dr.Jit targets compute_70.
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
    if(WIN32)
        get_filename_component(PYTHON_ENV_ROOT "${Python_EXECUTABLE}" DIRECTORY)
    endif()
    if(WIN32 AND NOT EXISTS "${TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
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
    if(DEFINED TORCH_CUDA_ARCH_LIST AND NOT TORCH_CUDA_ARCH_LIST STREQUAL "")
        # Explicit CMake configuration wins for cross-compilation.
    elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
        set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
    else()
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch; major, minor = torch.cuda.get_device_capability(); print(f'{major}.{minor}')"
            RESULT_VARIABLE TORCH_CUDA_ARCH_RESULT
            OUTPUT_VARIABLE TORCH_CUDA_ARCH_LIST
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT TORCH_CUDA_ARCH_RESULT EQUAL 0 OR TORCH_CUDA_ARCH_LIST STREQUAL "")
            message(FATAL_ERROR "Could not detect the local Torch CUDA architecture. Set CMAKE_CUDA_ARCHITECTURES and TORCH_CUDA_ARCH_LIST explicitly for cross-compilation.")
        endif()
    endif()
    find_package(Torch REQUIRED)
    if(NOT RAYD_TORCH_CUDA_GENCODE_FLAGS STREQUAL "")
        string(
            REGEX REPLACE
            "(^|[ \t])-gencode[ \t]+arch=[^ \t]+,code=[^ \t]+"
            ""
            CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS}")
        string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
        if(NOT RAYD_TORCH_CALLER_CUDA_FLAGS STREQUAL "")
            string(
                FIND
                "${CMAKE_CUDA_FLAGS}"
                "${RAYD_TORCH_CALLER_CUDA_FLAGS}"
                RAYD_TORCH_CALLER_CUDA_FLAGS_INDEX)
            if(RAYD_TORCH_CALLER_CUDA_FLAGS_INDEX EQUAL -1)
                string(
                    PREPEND
                    CMAKE_CUDA_FLAGS
                    "${RAYD_TORCH_CALLER_CUDA_FLAGS} ")
            endif()
        endif()
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    elseif(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${RAYD_TORCH_REQUESTED_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES "${RAYD_TORCH_REQUESTED_CUDA_ARCHITECTURES}")
    endif()
    if(RAYD_TORCH_CUDA_GENCODE_FLAGS STREQUAL ""
       AND (NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF"))
        message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES resolved to '${CMAKE_CUDA_ARCHITECTURES}'; RayD Torch kernels would build without -gencode flags.")
    endif()
    if(NOT RAYD_TORCH_CUDA_GENCODE_FLAGS STREQUAL "")
        message(STATUS "RayD Torch grouped CUDA gencode flags: ${RAYD_TORCH_CUDA_GENCODE_FLAGS}")
        message(STATUS "RayD Torch CUDA flags after removing Torch gencode flags: ${CMAKE_CUDA_FLAGS}")
    else()
        message(STATUS "RayD Torch CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
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
    add_library(
        rayd_torch_stable_ops
        SHARED
            ${RAYD_ROOT_DIR}/src/camera/camera_stable.cu
            ${RAYD_ROOT_DIR}/src/scene/intersection_stable.cu)
    set_target_properties(
        rayd_torch_stable_ops
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME "_stable_ops"
            POSITION_INDEPENDENT_CODE ON)
    rayd_torch_apply_cuda_gencode(rayd_torch_stable_ops)
    target_compile_definitions(
        rayd_torch_stable_ops
        PRIVATE
            TORCH_TARGET_VERSION=0x020a000000000000
            USE_CUDA)
    target_include_directories(
        rayd_torch_stable_ops
        PRIVATE
            ${RAYD_ROOT_DIR}
            ${RAYD_INCLUDE_DIR}
            "${TORCH_INSTALL_PREFIX}/include"
            ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(
        rayd_torch_stable_ops
        PRIVATE
            "${RAYD_TORCH_STABLE_CPU_LIBRARY}"
            "${RAYD_TORCH_STABLE_CUDA_LIBRARY}"
            CUDA::cudart)
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

    set(RAYD_TORCH_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_intersect.ptx")
    set(RAYD_TORCH_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/scene/intersection_torch_ptx.h")
    set(RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_point_ray.ptx")
    set(RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/edge/point_ray_torch_ptx.h")
    set(RAYD_TORCH_EDGE_OPTIX_TOPK_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix_topk.ptx")
    set(RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/edge/topk_torch_ptx.h")
    set(RAYD_TORCH_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_optix.ptx")
    set(RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/reflection/trace_torch_ptx.h")
    set(RAYD_TORCH_SEGMENT_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_visibility_optix.ptx")
    set(RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/visibility/segment_torch_ptx.h")
    set(RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/axial_edge_visibility_optix.ptx")
    set(RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/visibility/axial_edge_torch_ptx.h")
    set(RAYD_TORCH_REFLECTION_EPC_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_optix.ptx")
    set(RAYD_TORCH_REFLECTION_EPC_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/reflection/epc_torch_ptx.h")
    set(RAYD_TORCH_REFLECTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_accumulation_optix.ptx")
    set(RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/reflection/accumulation_torch_ptx.h")
    set(RAYD_TORCH_DIFFRACTION_PATHS_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_paths_optix.ptx")
    set(RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/diffraction/paths_torch_ptx.h")
    set(RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_optix.ptx")
    set(RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/diffraction/accumulation_torch_ptx.h")
    set(RAYD_TORCH_SEGMENT_PENETRATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_penetration_optix.ptx")
    set(RAYD_TORCH_SEGMENT_PENETRATION_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/rayd/penetration/segment_torch_ptx.h")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/scene/intersection_optix.cu"
            -o "${RAYD_TORCH_OPTIX_PTX}"
            -MD
            -MF "${RAYD_TORCH_OPTIX_PTX}.d"
            -MT "${RAYD_TORCH_OPTIX_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/scene/intersection_optix.cu
            ${RAYD_ROOT_DIR}/src/scene/optix_intersect_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/scene/optix_contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/scene/optix_device.cuh"
        DEPFILE "${RAYD_TORCH_OPTIX_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_OPTIX_PTX}"
            "${RAYD_TORCH_OPTIX_PTX_HEADER}"
            rayd_torch_optix_intersect_ptx
        DEPENDS
            "${RAYD_TORCH_OPTIX_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_optix_ptx DEPENDS "${RAYD_TORCH_OPTIX_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/edge/edge_optix.cu"
            -o "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            -MD
            -MF "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}.d"
            -MT "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            -DRAYD_TORCH_EDGE_POINT_RAY_ONLY=1
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/edge/edge_optix.cu
            ${RAYD_ROOT_DIR}/src/edge/optix_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/edge_distance.h"
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/optix_contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/optix_device.cuh"
        DEPFILE "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX_HEADER}"
            rayd_torch_edge_optix_point_ray_ptx
        DEPENDS
            "${RAYD_TORCH_EDGE_OPTIX_POINT_RAY_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/edge/edge_optix.cu"
            -o "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            -MD
            -MF "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}.d"
            -MT "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            -DRAYD_TORCH_EDGE_TOPK_ONLY=1
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/edge/edge_optix.cu
            ${RAYD_ROOT_DIR}/src/edge/optix_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/edge_distance.h"
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/optix_contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/edge/optix_device.cuh"
        DEPFILE "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX_HEADER}"
            rayd_torch_edge_optix_topk_ptx
        DEPENDS
            "${RAYD_TORCH_EDGE_OPTIX_TOPK_PTX}"
            scripts/embed_ptx.py
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
            "${RAYD_ROOT_DIR}/src/reflection/trace_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            -MD
            -MF "${RAYD_TORCH_REFLECTION_TRACE_PTX}.d"
            -MT "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/reflection/trace_optix.cu
            ${RAYD_ROOT_DIR}/src/reflection/trace_params.h
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/reflection_geometry.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/optix_hit.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/optix_traverser.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/trace_params.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/trace_optix_device.cuh"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/trace_algo.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/qualifiers.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/traverser.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/numeric_policy.h"
        DEPFILE "${RAYD_TORCH_REFLECTION_TRACE_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}"
            rayd_torch_reflection_trace_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_TRACE_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_trace_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_TRACE_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/visibility/visibility_optix.cu"
            -o "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            -MD
            -MF "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}.d"
            -MT "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/visibility/visibility_optix.cu
            ${RAYD_ROOT_DIR}/src/visibility/visibility_params.h
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_params.h"
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_optix_device.cuh"
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_algo.h"
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/qualifiers.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/traverser.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/numeric_policy.h"
        DEPFILE "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}"
            rayd_torch_segment_visibility_optix_ptx
        DEPENDS
            "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_segment_visibility_optix_ptx DEPENDS "${RAYD_TORCH_SEGMENT_VISIBILITY_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/visibility/axial_edge_visibility_optix.cu"
            -o "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"
            -MD
            -MF "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}.d"
            -MT "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/visibility/axial_edge_visibility_optix.cu
            ${RAYD_ROOT_DIR}/src/visibility/axial_edge_visibility_params.h
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_algo.h"
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_optix_device.cuh"
            "${RAYD_INCLUDE_DIR}/rayd/visibility/segment_params.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/numeric_policy.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/qualifiers.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/traverser.h"
        DEPFILE "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"
            "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX_HEADER}"
            rayd_torch_axial_edge_visibility_optix_ptx
        DEPENDS
            "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(
        rayd_torch_axial_edge_visibility_optix_ptx
        DEPENDS "${RAYD_TORCH_AXIAL_EDGE_VISIBILITY_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_EPC_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/reflection/epc_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            -MD
            -MF "${RAYD_TORCH_REFLECTION_EPC_PTX}.d"
            -MT "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/reflection/epc_optix.cu
            ${RAYD_ROOT_DIR}/src/reflection/epc_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/optix_hit.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/epc_params.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/epc_optix_device.cuh"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/epc_algo.h"
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/reflection_geometry.h"
            "${RAYD_INCLUDE_DIR}/rayd/reflection/epc_chain.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/qualifiers.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/traverser.h"
            "${RAYD_INCLUDE_DIR}/rayd/rt/numeric_policy.h"
        DEPFILE "${RAYD_TORCH_REFLECTION_EPC_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}"
            rayd_torch_reflection_epc_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_EPC_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_epc_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_EPC_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/reflection/accumulation_optix.cu"
            -o "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            -MD
            -MF "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}.d"
            -MT "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/reflection/accumulation_optix.cu
            ${RAYD_INCLUDE_DIR}/rayd/math.h
            ${RAYD_ROOT_DIR}/src/reflection/accum_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
            "${RAYD_INCLUDE_DIR}/rayd/math.h"
        DEPFILE "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}"
            rayd_torch_reflection_accumulation_optix_ptx
        DEPENDS
            "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_reflection_accumulation_optix_ptx DEPENDS "${RAYD_TORCH_REFLECTION_ACCUMULATION_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/diffraction/paths_optix.cu"
            -o "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            -MD
            -MF "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}.d"
            -MT "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/diffraction/paths_optix.cu
            ${RAYD_ROOT_DIR}/src/diffraction/common.h
            ${RAYD_ROOT_DIR}/src/diffraction/paths_params.h
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/utd.h"
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/utd_types.h"
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/contracts.h"
        DEPFILE "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}"
            rayd_torch_diffraction_paths_optix_ptx
        DEPENDS
            "${RAYD_TORCH_DIFFRACTION_PATHS_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_diffraction_paths_optix_ptx DEPENDS "${RAYD_TORCH_DIFFRACTION_PATHS_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/diffraction/accumulation_optix.cu"
            -o "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            -MD
            -MF "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}.d"
            -MT "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            ${RAYD_TORCH_OPTIX_NVCC_FLAGS}
        DEPENDS
            ${RAYD_ROOT_DIR}/src/diffraction/accumulation_optix.cu
            ${RAYD_ROOT_DIR}/src/diffraction/common.h
            ${RAYD_ROOT_DIR}/src/diffraction/accum_params.h
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/utd.h"
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/utd_types.h"
            "${RAYD_INCLUDE_DIR}/rayd/diffraction/contracts.h"
        DEPFILE "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}"
            rayd_torch_diffraction_accumulation_optix_ptx
        DEPENDS
            "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(rayd_torch_diffraction_accumulation_optix_ptx DEPENDS "${RAYD_TORCH_DIFFRACTION_ACCUMULATION_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${RAYD_ROOT_DIR}/src/penetration/penetration_optix.cu"
            -o "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"
            -MD
            -MF "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}.d"
            -MT "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"
            -I "${RAYD_ROOT_DIR}"
            -I "${RAYD_INCLUDE_DIR}"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            --std=c++17
            --gpu-architecture=compute_75
            --ftz=false
            --prec-div=true
            --prec-sqrt=true
        DEPENDS
            ${RAYD_ROOT_DIR}/src/penetration/penetration_optix.cu
            ${RAYD_ROOT_DIR}/src/penetration/segment_penetration_params.h
            "${RAYD_INCLUDE_DIR}/rayd/contracts.h"
        DEPFILE "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}.d"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${RAYD_TORCH_SEGMENT_PENETRATION_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_ptx.py"
            "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"
            "${RAYD_TORCH_SEGMENT_PENETRATION_PTX_HEADER}"
            rayd_torch_segment_penetration_optix_ptx
        DEPENDS
            "${RAYD_TORCH_SEGMENT_PENETRATION_PTX}"
            scripts/embed_ptx.py
    )
    add_custom_target(
        rayd_torch_segment_penetration_optix_ptx
        DEPENDS "${RAYD_TORCH_SEGMENT_PENETRATION_PTX_HEADER}")

    # ADR-0035 profile: nvcc_default for every CUDA source below that has no
    # COMPILE_OPTIONS override in the three blocks that follow the list.
    set(
        RAYD_TORCH_NATIVE_CORE_SOURCES
        ${RAYD_ROOT_DIR}/src/bindings/tensor_contract.cpp
        ${RAYD_ROOT_DIR}/src/runtime/optix.cpp
        ${RAYD_ROOT_DIR}/src/runtime/diagnostics.cpp
        ${RAYD_ROOT_DIR}/src/camera/camera.cpp
        ${RAYD_ROOT_DIR}/src/scene/scene.cpp
        ${RAYD_ROOT_DIR}/src/scene/intersection.cpp
        ${RAYD_ROOT_DIR}/src/edge/edge.cpp
        ${RAYD_ROOT_DIR}/src/visibility/visibility.cpp
        ${RAYD_ROOT_DIR}/src/reflection/reflection.cpp
        ${RAYD_ROOT_DIR}/src/diffraction/diffraction.cpp
        ${RAYD_ROOT_DIR}/src/penetration/penetration.cpp
        ${RAYD_ROOT_DIR}/src/sdf/sdf.cpp
        ${RAYD_ROOT_DIR}/src/scene/cache.cu
        ${RAYD_ROOT_DIR}/src/scene/packing_shared.cu
        ${RAYD_ROOT_DIR}/src/scene/intersection.cu
        ${RAYD_ROOT_DIR}/src/scene/triangle_bvh.cu
        ${RAYD_ROOT_DIR}/src/scene/multipath.cu
        ${RAYD_ROOT_DIR}/src/penetration/penetration.cu
        ${RAYD_ROOT_DIR}/src/bvh/build_shared.cu
        ${RAYD_ROOT_DIR}/src/bvh/triangle_query_shared.cu
        ${RAYD_ROOT_DIR}/src/edge/edge_shared.cu
        ${RAYD_ROOT_DIR}/src/edge/edge_bvh.cu
        ${RAYD_ROOT_DIR}/src/edge/edge_queries.cu
        ${RAYD_ROOT_DIR}/src/visibility/visibility_ad.cu
        ${RAYD_ROOT_DIR}/src/reflection/reflection_kernels.cu
        ${RAYD_ROOT_DIR}/src/reflection/dedup_shared.cu
        ${RAYD_ROOT_DIR}/src/transmission/transmission.cu
        ${RAYD_ROOT_DIR}/src/diffraction/wedge.cu
        ${RAYD_ROOT_DIR}/src/scattering/table.cu
        ${RAYD_ROOT_DIR}/src/scattering/table_ad.cu
        ${RAYD_ROOT_DIR}/src/scattering/ensemble.cu
        ${RAYD_ROOT_DIR}/src/scattering/patch.cu
        ${RAYD_ROOT_DIR}/src/scattering/chain_ensemble.cu
        ${RAYD_ROOT_DIR}/src/scattering/chain_realization.cu
        ${RAYD_ROOT_DIR}/src/diffraction/diffraction_kernels.cu
        ${RAYD_ROOT_DIR}/src/runtime/diagnostics.cu
        ${RAYD_ROOT_DIR}/src/camera/camera.cu
        ${RAYD_ROOT_DIR}/src/sdf/sdf.cu
    )

    # ADR-0035 profile: fast_math, and divergence D3: this is the only Torch
    # consumer of the shared UTD math compiled that way, while the fused
    # multipath executor reads the same headers at nvcc defaults and the nine
    # scattering units below read them without FMA contraction. No family record
    # explains the choice; it is recorded as uncontracted, not as approved.
    set_source_files_properties(
        ${RAYD_ROOT_DIR}/src/diffraction/wedge.cu
        PROPERTIES
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

    # ADR-0033 freezes the two penetration policy expressions and their native
    # VJP/JVP companions to precise division/square-root and non-FTZ behavior,
    # while retaining the established default FMA-contraction behavior. The
    # standalone OptiX PTX command above carries the same family-local policy.
    # ADR-0035 profile: precise_no_ftz, family-exhaustive. These three flags are
    # nvcc's own defaults; they are spelled out so the family cannot inherit a
    # fast-math option added elsewhere.
    set_source_files_properties(
        ${RAYD_ROOT_DIR}/src/penetration/penetration.cu
        PROPERTIES
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--ftz=false;--prec-div=true;--prec-sqrt=true>")

    # ADR-0026 preserves the source-local arithmetic policy from Channel. The
    # sampling/eval primal in scattering.cu deliberately keeps target defaults;
    # the lockstep/AD translation units disable FMA contraction.
    # ADR-0035 profile: no_fmad, family-exhaustive. No other translation unit in
    # either backend may take this profile.
    set_source_files_properties(
        ${RAYD_ROOT_DIR}/src/scattering/table_ad.cu
        ${RAYD_ROOT_DIR}/src/scattering/ensemble.cu
        ${RAYD_ROOT_DIR}/src/scattering/patch.cu
        ${RAYD_ROOT_DIR}/src/scattering/chain_ensemble.cu
        ${RAYD_ROOT_DIR}/src/scattering/chain_realization.cu
        PROPERTIES
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

    # Exactly these eight units reach torch/extension.h, and through it pybind11
    # and the CPython C API, so only they are Python-version dependent. The rest
    # -- every CUDA translation unit and four more C++ units -- compile
    # identically for every supported CPython version. Keeping the two halves in
    # separate static libraries lets a build produce the Python-free half once
    # and reuse it across the per-version wheel jobs. `rayd_torch_native_core`
    # stays the name every consumer and the ADR-0035 contract parser already use.
    set(RAYD_TORCH_PYTHON_OPS_SOURCES
        ${RAYD_ROOT_DIR}/src/runtime/diagnostics.cpp
        ${RAYD_ROOT_DIR}/src/camera/camera.cpp
        ${RAYD_ROOT_DIR}/src/scene/scene.cpp
        ${RAYD_ROOT_DIR}/src/scene/intersection.cpp
        ${RAYD_ROOT_DIR}/src/edge/edge.cpp
        ${RAYD_ROOT_DIR}/src/visibility/visibility.cpp
        ${RAYD_ROOT_DIR}/src/reflection/reflection.cpp
        ${RAYD_ROOT_DIR}/src/diffraction/diffraction.cpp)
    set(RAYD_TORCH_DEVICE_CORE_SOURCES ${RAYD_TORCH_NATIVE_CORE_SOURCES})
    list(REMOVE_ITEM RAYD_TORCH_DEVICE_CORE_SOURCES ${RAYD_TORCH_PYTHON_OPS_SOURCES})
    list(LENGTH RAYD_TORCH_NATIVE_CORE_SOURCES RAYD_TORCH_CORE_SOURCE_COUNT)
    list(LENGTH RAYD_TORCH_DEVICE_CORE_SOURCES RAYD_TORCH_DEVICE_SOURCE_COUNT)
    math(EXPR RAYD_TORCH_REMOVED_COUNT
        "${RAYD_TORCH_CORE_SOURCE_COUNT} - ${RAYD_TORCH_DEVICE_SOURCE_COUNT}")
    if(NOT RAYD_TORCH_REMOVED_COUNT EQUAL 8)
        message(FATAL_ERROR
            "Expected to split 8 Python-coupled units out of "
            "RAYD_TORCH_NATIVE_CORE_SOURCES, removed ${RAYD_TORCH_REMOVED_COUNT}. "
            "A renamed or dropped source silently leaves it in the Python-free "
            "half; re-audit the torch/extension.h include closure.")
    endif()

    add_library(rayd_torch_device_core STATIC ${RAYD_TORCH_DEVICE_CORE_SOURCES})
    add_library(rayd_torch_python_ops STATIC ${RAYD_TORCH_PYTHON_OPS_SOURCES})
    add_library(rayd_torch_native_core INTERFACE)
    target_link_libraries(
        rayd_torch_native_core
        INTERFACE rayd_torch_python_ops rayd_torch_device_core)
    set_target_properties(
        rayd_torch_device_core rayd_torch_python_ops
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
    rayd_torch_apply_cuda_gencode(rayd_torch_device_core)
    rayd_torch_apply_cuda_gencode(rayd_torch_python_ops)
    # The Python-free half must not see Python.h: losing that include directory
    # is what makes it reusable across CPython versions.
    target_include_directories(
        rayd_torch_device_core
        PUBLIC
            ${RAYD_INCLUDE_DIR}
        PRIVATE
            ${RAYD_ROOT_DIR}
            "${TORCH_INSTALL_PREFIX}/include"
            "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    target_include_directories(
        rayd_torch_python_ops
        PUBLIC
            ${RAYD_INCLUDE_DIR}
        PRIVATE
            ${RAYD_ROOT_DIR}
            ${Python_INCLUDE_DIRS}
            "${TORCH_INSTALL_PREFIX}/include"
            "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    add_dependencies(
        rayd_torch_device_core
        rayd_torch_optix_ptx
        rayd_torch_edge_optix_ptx
        rayd_torch_reflection_trace_optix_ptx
        rayd_torch_segment_visibility_optix_ptx
        rayd_torch_axial_edge_visibility_optix_ptx
        rayd_torch_reflection_epc_optix_ptx
        rayd_torch_reflection_accumulation_optix_ptx
        rayd_torch_diffraction_paths_optix_ptx
        rayd_torch_diffraction_accumulation_optix_ptx
        rayd_torch_segment_penetration_optix_ptx)
    add_dependencies(rayd_torch_python_ops rayd_torch_device_core)
    target_link_libraries(
        rayd_torch_device_core
        PUBLIC
            "${TORCH_LIBRARIES}"
            CUDA::cudart
            CUDA::cuda_driver)
    # libtorch_python is the Python-version dependent half of LibTorch and is
    # needed only by the units that reach torch/extension.h.
    target_link_libraries(
        rayd_torch_python_ops
        PUBLIC
            "${TORCH_LIBRARIES}"
            "${TORCH_PYTHON_LIBRARY}"
            CUDA::cudart
            CUDA::cuda_driver)
    target_compile_features(rayd_torch_device_core PUBLIC cxx_std_17)
    target_compile_features(rayd_torch_python_ops PUBLIC cxx_std_17)

    if(BUILD_TESTING AND RAYD_TORCH_BUILD_CPP_TESTS)
        add_executable(
            rayd_torch_integration_test
            ${RAYD_ROOT_DIR}/tests/native/integration_test.cpp)
        target_include_directories(
            rayd_torch_integration_test
            PRIVATE
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(
            rayd_torch_integration_test
            PRIVATE
                rayd_torch_native_core
                Python::Python)
        target_compile_features(rayd_torch_integration_test PRIVATE cxx_std_17)
        add_test(
            NAME rayd_torch_integration
            COMMAND rayd_torch_integration_test)
        add_executable(
            rayd_torch_segment_penetration_test
            ${RAYD_ROOT_DIR}/tests/penetration/segment_penetration_test.cpp
            ${RAYD_ROOT_DIR}/tests/penetration/segment_penetration_oracle.cu)
        target_include_directories(
            rayd_torch_segment_penetration_test
            PRIVATE
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(
            rayd_torch_segment_penetration_test
            PRIVATE
                rayd_torch_native_core
                Python::Python)
        target_compile_features(
            rayd_torch_segment_penetration_test PRIVATE cxx_std_17)
        set_target_properties(
            rayd_torch_segment_penetration_test
            PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        add_test(
            NAME rayd_torch_segment_penetration
            COMMAND rayd_torch_segment_penetration_test)
        add_executable(
            rayd_torch_diffraction_wedge_test
            ${RAYD_ROOT_DIR}/tests/diffraction/diffraction_wedge_test.cpp)
        target_include_directories(
            rayd_torch_diffraction_wedge_test
            PRIVATE
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(
            rayd_torch_diffraction_wedge_test
            PRIVATE
                rayd_torch_native_core
                Python::Python)
        target_compile_features(
            rayd_torch_diffraction_wedge_test PRIVATE cxx_std_17)
        add_test(
            NAME rayd_torch_diffraction_wedge
            COMMAND rayd_torch_diffraction_wedge_test)
        add_executable(
            rayd_torch_scattering_test
            ${RAYD_ROOT_DIR}/tests/scattering/scattering_test.cpp)
        target_include_directories(
            rayd_torch_scattering_test
            PRIVATE
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(
            rayd_torch_scattering_test
            PRIVATE
                rayd_torch_native_core
                Python::Python)
        target_compile_features(
            rayd_torch_scattering_test PRIVATE cxx_std_17)
        add_test(
            NAME rayd_torch_scattering
            COMMAND rayd_torch_scattering_test)
        add_executable(
            rayd_torch_scattering_chain_test
            ${RAYD_ROOT_DIR}/tests/scattering/scattering_chain_test.cpp)
        target_include_directories(
            rayd_torch_scattering_chain_test
            PRIVATE
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(
            rayd_torch_scattering_chain_test
            PRIVATE
                rayd_torch_native_core
                Python::Python)
        target_compile_features(
            rayd_torch_scattering_chain_test PRIVATE cxx_std_17)
        add_test(
            NAME rayd_torch_scattering_chain
            COMMAND rayd_torch_scattering_chain_test)
        if(WIN32)
            set_tests_properties(
                rayd_torch_integration
                rayd_torch_segment_penetration
                rayd_torch_diffraction_wedge
                rayd_torch_scattering
                rayd_torch_scattering_chain
                PROPERTIES
                    ENVIRONMENT_MODIFICATION
                        "PATH=path_list_prepend:${TORCH_INSTALL_PREFIX}/lib;PATH=path_list_prepend:${PYTHON_ENV_ROOT};PATH=path_list_prepend:${PYTHON_ENV_ROOT}/Library/bin;PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
        endif()
    endif()

    if(RAYD_TORCH_BUILD_PYTHON_MODULE)
        add_library(
            rayd_torch_legacy_ops
            SHARED
                ${RAYD_ROOT_DIR}/src/bindings/library.cpp
                ${RAYD_ROOT_DIR}/src/bindings/legacy_anchor.cpp)
        set_target_properties(
            rayd_torch_legacy_ops
            PROPERTIES
                PREFIX ""
                OUTPUT_NAME "_legacy_ops"
                POSITION_INDEPENDENT_CODE ON)
        target_include_directories(
            rayd_torch_legacy_ops
            PRIVATE
                ${RAYD_ROOT_DIR}
                ${RAYD_INCLUDE_DIR}
                ${Python_INCLUDE_DIRS}
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
                ${CMAKE_CURRENT_BINARY_DIR}/generated
                ${CUDAToolkit_INCLUDE_DIRS}
                ${OPTIX_INCLUDE_DIR}
        )
        target_link_libraries(
            rayd_torch_legacy_ops
            PRIVATE
                rayd_torch_native_core
                Python::Module)
        target_compile_features(rayd_torch_legacy_ops PRIVATE cxx_std_17)
        install(
            TARGETS rayd_torch_legacy_ops
            LIBRARY DESTINATION rayd/torch
            RUNTIME DESTINATION rayd/torch)

        find_path(
            RAYD_TORCH_PYBIND11_INCLUDE_DIR
            pybind11/pybind11.h
            HINTS "${TORCH_INSTALL_PREFIX}/include"
            REQUIRED)
        Python_add_library(_C MODULE WITH_SOABI ${RAYD_ROOT_DIR}/src/bindings/module.cpp)
        target_include_directories(
            _C
            PRIVATE
                ${RAYD_INCLUDE_DIR}
                ${RAYD_TORCH_PYBIND11_INCLUDE_DIR})
        target_compile_features(_C PRIVATE cxx_std_17)
        install(TARGETS _C LIBRARY DESTINATION rayd/torch RUNTIME DESTINATION rayd/torch)
    endif()
endif()
