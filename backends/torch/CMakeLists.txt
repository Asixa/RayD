cmake_minimum_required(VERSION 3.26)
project(raydtorch LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

option(RAYDTORCH_BUILD_NATIVE "Build the RayDTorch extension." ON)

if(RAYDTORCH_BUILD_NATIVE)
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

    function(pybind11_add_module target)
        Python_add_library(${target} MODULE WITH_SOABI ${ARGN})
        target_include_directories(
            ${target}
            PRIVATE
                "${TORCH_INSTALL_PREFIX}/include"
                "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
        )
    endfunction()

    pybind11_add_module(_raydtorch src/torch_ext/module.cpp src/torch_ext/tensor_check.cpp)
    target_include_directories(_raydtorch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(_raydtorch PRIVATE "${TORCH_LIBRARIES}")
    target_compile_features(_raydtorch PRIVATE cxx_std_17)
    install(TARGETS _raydtorch LIBRARY DESTINATION raydtorch RUNTIME DESTINATION raydtorch)
endif()
