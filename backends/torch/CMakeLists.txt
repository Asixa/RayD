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

    pybind11_add_module(
        _raydtorch
        src/torch_ext/module.cpp
        src/torch_ext/tensor_check.cpp
        src/torch_ext/scene_cache.cpp
        src/torch_ext/ops_scene.cpp
    )
    target_include_directories(
        _raydtorch
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${OPTIX_INCLUDE_DIR}
    )
    target_link_libraries(_raydtorch PRIVATE "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)
    target_compile_features(_raydtorch PRIVATE cxx_std_17)
    install(TARGETS _raydtorch LIBRARY DESTINATION raydtorch RUNTIME DESTINATION raydtorch)
endif()
