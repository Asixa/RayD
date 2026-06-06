cmake_minimum_required(VERSION 3.26)
project(raydtorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(RAYDTORCH_BUILD_NATIVE "Build the RayDTorch extension." ON)

if(RAYDTORCH_BUILD_NATIVE)
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
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

    function(pybind11_add_module target)
        Python_add_library(${target} MODULE WITH_SOABI ${ARGN})
        target_include_directories(
            ${target}
            PRIVATE
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
        )
    endfunction()

    set(RAYDTORCH_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/optix_intersect.ptx")
    set(RAYDTORCH_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/optix_intersect_ptx.h")
    set(RAYDTORCH_EDGE_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix.ptx")
    set(RAYDTORCH_EDGE_OPTIX_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/raydtorch/edge_optix_ptx.h")
    add_custom_command(
        OUTPUT "${RAYDTORCH_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/kernels/optix_intersect.cu"
            -o "${RAYDTORCH_OPTIX_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            --std=c++17
        DEPENDS
            src/torch_ext/kernels/optix_intersect.cu
            include/raydtorch/optix_intersect_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_OPTIX_PTX}"
            "${RAYDTORCH_OPTIX_PTX_HEADER}"
            raydtorch_optix_intersect_ptx
        DEPENDS
            "${RAYDTORCH_OPTIX_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_optix_ptx DEPENDS "${RAYDTORCH_OPTIX_PTX_HEADER}")
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_PTX}"
        COMMAND
            "${CMAKE_CUDA_COMPILER}"
            --ptx
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/kernels/edge_optix.cu"
            -o "${RAYDTORCH_EDGE_OPTIX_PTX}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
            -I "${CUDAToolkit_INCLUDE_DIRS}"
            -I "${OPTIX_INCLUDE_DIR}"
            --std=c++17
        DEPENDS
            src/torch_ext/kernels/edge_optix.cu
            include/raydtorch/edge_optix_params.h
    )
    add_custom_command(
        OUTPUT "${RAYDTORCH_EDGE_OPTIX_PTX_HEADER}"
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_ext/embed_ptx.py"
            "${RAYDTORCH_EDGE_OPTIX_PTX}"
            "${RAYDTORCH_EDGE_OPTIX_PTX_HEADER}"
            raydtorch_edge_optix_ptx
        DEPENDS
            "${RAYDTORCH_EDGE_OPTIX_PTX}"
            src/torch_ext/embed_ptx.py
    )
    add_custom_target(raydtorch_edge_optix_ptx DEPENDS "${RAYDTORCH_EDGE_OPTIX_PTX_HEADER}")

    pybind11_add_module(
        _raydtorch
        src/torch_ext/module.cpp
        src/torch_ext/tensor_check.cpp
        src/torch_ext/scene_cache.cpp
        src/torch_ext/ops_scene.cpp
        src/torch_ext/optix_context.cpp
        src/torch_ext/ops_intersect.cpp
        src/torch_ext/ops_edge.cpp
        src/torch_ext/kernels/geometry_forward.cu
        src/torch_ext/kernels/geometry_backward.cu
        src/torch_ext/kernels/edge_forward.cu
        src/torch_ext/kernels/edge_backward.cu
    )
    target_include_directories(
        _raydtorch
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/generated
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    add_dependencies(_raydtorch raydtorch_optix_ptx raydtorch_edge_optix_ptx)
    target_link_libraries(_raydtorch PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart CUDA::cuda_driver)
    target_compile_features(_raydtorch PRIVATE cxx_std_17)
    install(TARGETS _raydtorch LIBRARY DESTINATION raydtorch RUNTIME DESTINATION raydtorch)
endif()
