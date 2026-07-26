cmake_minimum_required(VERSION 3.15...3.22)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RAYD_OPTIX_PRODUCTION_CONFIG "Use OptiX production module settings (LEVEL_3 + no exceptions)." ON)
option(RAYD_NANOBIND_NOMINSIZE "Disable nanobind's size-oriented optimization flags in optimized configs." ON)
option(RAYD_ENABLE_LTO "Enable nanobind/C++ link-time optimization." OFF)
option(RAYD_REGENERATE_REFLECTION_TRACE_PTX "Regenerate reflection_trace PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX "Regenerate reflection_accumulation PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX "Regenerate diffraction_accumulation PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_DIFFRACTION_PATHS_PTX "Regenerate diffraction_paths PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_REFLECTION_EPC_PTX "Regenerate reflection_epc PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX "Regenerate segment_visibility PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_SURFEL_TRACE_PTX "Regenerate surfel trace PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_REGENERATE_EDGE_OPTIX_PTX "Regenerate edge OptiX custom-AABB PTX during the build. Requires OptiX SDK headers." OFF)
option(RAYD_STRICT_PTX_SOURCE_CHECK "Fail configuration when the committed PTX source-identity record is stale." OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# The committed *_ptx.h headers are regenerated only on request, so an edit to a
# .cu file or to any header it reaches otherwise leaves them describing older
# device code. Replay the source-identity audit that
# tests/test_ptx_source_digest.py enforces. That test is the authority; this is
# the early warning for a developer who is mid-edit. It stays a warning by
# default because regenerating PTX needs an OptiX SDK that a source build is not
# required to have; release configures set RAYD_STRICT_PTX_SOURCE_CHECK.
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/audit_ptx_sources.py" --check
    RESULT_VARIABLE RAYD_PTX_AUDIT_RESULT
    OUTPUT_VARIABLE RAYD_PTX_AUDIT_OUTPUT
    ERROR_VARIABLE RAYD_PTX_AUDIT_ERROR
)
if(NOT RAYD_PTX_AUDIT_RESULT EQUAL 0)
    if(RAYD_STRICT_PTX_SOURCE_CHECK)
        message(FATAL_ERROR "${RAYD_PTX_AUDIT_ERROR}")
    else()
        message(WARNING
            "${RAYD_PTX_AUDIT_ERROR}\n"
            "Configure with -DRAYD_STRICT_PTX_SOURCE_CHECK=ON to make this fatal.")
    endif()
endif()
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib, sys; spec = importlib.util.find_spec('drjit'); sys.stdout.write(str(pathlib.Path(next(iter(spec.submodule_search_locations))).resolve() / 'cmake')) if spec and spec.submodule_search_locations else sys.exit(1)"
    RESULT_VARIABLE DRJIT_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE DRJIT_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT DRJIT_CMAKE_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed Dr.Jit CMake package. Ensure drjit>=1.3.1 is installed in the build environment.")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib, sys; spec = importlib.util.find_spec('nanobind'); sys.stdout.write(str(pathlib.Path(next(iter(spec.submodule_search_locations))).resolve() / 'cmake')) if spec and spec.submodule_search_locations else sys.exit(1)"
    RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed nanobind CMake package.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${DRJIT_CMAKE_DIR}" "${NANOBIND_CMAKE_DIR}")
find_package(drjit CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)

get_target_property(RAYD_DRJIT_INCLUDE_DIRS drjit INTERFACE_INCLUDE_DIRECTORIES)
if(NOT RAYD_DRJIT_INCLUDE_DIRS)
    set(RAYD_DRJIT_INCLUDE_DIRS "")
endif()
get_target_property(RAYD_DRJIT_COMPILE_DEFINITIONS drjit-core INTERFACE_COMPILE_DEFINITIONS)
if(NOT RAYD_DRJIT_COMPILE_DEFINITIONS)
    set(RAYD_DRJIT_COMPILE_DEFINITIONS "")
endif()

if(WIN32)
    add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-reorder -Wno-sign-compare -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -DNDEBUG")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(RAYD_OPTIX_PRODUCTION_CONFIG)
    set(RAYD_OPTIX_MODULE_OPT_LEVEL 0x2343)
    set(RAYD_OPTIX_EXCEPTION_FLAGS 0)
else()
    set(RAYD_OPTIX_MODULE_OPT_LEVEL 0x2340)
    set(RAYD_OPTIX_EXCEPTION_FLAGS 11)
endif()
if(NOT DEFINED RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL)
    set(RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL ${RAYD_OPTIX_MODULE_OPT_LEVEL})
endif()
if(NOT DEFINED RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS)
    set(RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS 11)
endif()

find_package(CUDA 11.0 REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

# FindCUDA does not support CMAKE_CUDA_COMPILER_LAUNCHER for the explicit NVCC
# custom commands below. CI may provide a compiler launcher through a small
# generated wrapper; local builds keep invoking NVCC directly.
set(RAYD_NVCC_LAUNCHER "" CACHE STRING
    "Optional compiler cache or launcher placed before NVCC.")
if(DEFINED ENV{RAYD_NVCC_LAUNCHER}
   AND NOT "$ENV{RAYD_NVCC_LAUNCHER}" STREQUAL "")
    set(RAYD_NVCC_LAUNCHER "$ENV{RAYD_NVCC_LAUNCHER}")
endif()
if(NOT RAYD_NVCC_LAUNCHER STREQUAL "")
    set(RAYD_REAL_NVCC_EXECUTABLE "${CUDA_NVCC_EXECUTABLE}")
    if(WIN32)
        set(RAYD_NVCC_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/rayd_nvcc.cmd")
        file(WRITE "${RAYD_NVCC_WRAPPER}"
            "@\"${RAYD_NVCC_LAUNCHER}\" \"${RAYD_REAL_NVCC_EXECUTABLE}\" %*\r\n")
    else()
        set(RAYD_NVCC_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/rayd_nvcc")
        file(WRITE "${RAYD_NVCC_WRAPPER}"
            "#!/bin/sh\nexec \"${RAYD_NVCC_LAUNCHER}\" \"${RAYD_REAL_NVCC_EXECUTABLE}\" \"$@\"\n")
        file(CHMOD "${RAYD_NVCC_WRAPPER}"
            PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
    endif()
    set(CUDA_NVCC_EXECUTABLE "${RAYD_NVCC_WRAPPER}")
    message(STATUS "RayD NVCC launcher: ${RAYD_NVCC_LAUNCHER}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(RAYD_SHARED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include")

set(RAYD_CUDA_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${RAYD_SHARED_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${CUDA_INCLUDE_DIRS}
    ${RAYD_DRJIT_INCLUDE_DIRS}
)
list(REMOVE_DUPLICATES RAYD_CUDA_INCLUDE_DIRS)

set(RAYD_CUDA_INCLUDE_FLAGS_LIST "")
foreach(RAYD_CUDA_INCLUDE_DIR IN LISTS RAYD_CUDA_INCLUDE_DIRS)
    list(APPEND RAYD_CUDA_INCLUDE_FLAGS_LIST "-I\"${RAYD_CUDA_INCLUDE_DIR}\"")
endforeach()
string(JOIN " " RAYD_CUDA_INCLUDE_FLAGS ${RAYD_CUDA_INCLUDE_FLAGS_LIST})

set(RAYD_CUDA_DEFINITIONS_LIST "")
foreach(RAYD_CUDA_DEFINITION IN LISTS RAYD_DRJIT_COMPILE_DEFINITIONS)
    list(APPEND RAYD_CUDA_DEFINITIONS_LIST "-D${RAYD_CUDA_DEFINITION}")
endforeach()
string(JOIN " " RAYD_CUDA_DEFINITIONS ${RAYD_CUDA_DEFINITIONS_LIST})

# Release wheels pass these through the environment because the CUDA sources
# are compiled by explicit nvcc custom commands rather than CMake's CUDA
# language. Local source builds may leave them empty and use nvcc's default.
set(RAYD_CUDA_GENCODE_ARCHES "" CACHE STRING
    "Comma-separated CUDA architectures that receive native SASS (for example 75,80,120).")
set(RAYD_CUDA_PTX_ARCH "" CACHE STRING
    "CUDA architecture that additionally receives forward-compatible PTX.")
if(DEFINED ENV{RAYD_CUDA_GENCODE_ARCHES} AND NOT "$ENV{RAYD_CUDA_GENCODE_ARCHES}" STREQUAL "")
    set(RAYD_CUDA_GENCODE_ARCHES "$ENV{RAYD_CUDA_GENCODE_ARCHES}")
endif()
if(DEFINED ENV{RAYD_CUDA_PTX_ARCH} AND NOT "$ENV{RAYD_CUDA_PTX_ARCH}" STREQUAL "")
    set(RAYD_CUDA_PTX_ARCH "$ENV{RAYD_CUDA_PTX_ARCH}")
endif()
if(RAYD_CUDA_GENCODE_ARCHES STREQUAL "")
    find_program(RAYD_NVIDIA_SMI_EXECUTABLE NAMES nvidia-smi)
    if(RAYD_NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND "${RAYD_NVIDIA_SMI_EXECUTABLE}" --query-gpu=compute_cap --format=csv,noheader
            RESULT_VARIABLE RAYD_NVIDIA_SMI_RESULT
            OUTPUT_VARIABLE RAYD_NVIDIA_SMI_COMPUTE_CAPABILITIES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(RAYD_NVIDIA_SMI_RESULT EQUAL 0)
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" RAYD_LOCAL_COMPUTE_CAPABILITY
                "${RAYD_NVIDIA_SMI_COMPUTE_CAPABILITIES}")
            if(NOT RAYD_LOCAL_COMPUTE_CAPABILITY STREQUAL "")
                string(REPLACE "." "" RAYD_CUDA_GENCODE_ARCHES
                    "${RAYD_LOCAL_COMPUTE_CAPABILITY}")
                message(STATUS
                    "RayD local CUDA architecture: sm_${RAYD_CUDA_GENCODE_ARCHES}")
            endif()
        endif()
    endif()
endif()
string(REPLACE "," ";" RAYD_CUDA_GENCODE_ARCH_LIST "${RAYD_CUDA_GENCODE_ARCHES}")
set(RAYD_CUDA_GENCODE_FLAGS_LIST "")
foreach(RAYD_CUDA_ARCH IN LISTS RAYD_CUDA_GENCODE_ARCH_LIST)
    string(STRIP "${RAYD_CUDA_ARCH}" RAYD_CUDA_ARCH)
    if(RAYD_CUDA_ARCH STREQUAL "")
        continue()
    endif()
    if(NOT RAYD_CUDA_ARCH MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Invalid CUDA architecture '${RAYD_CUDA_ARCH}' in RAYD_CUDA_GENCODE_ARCHES.")
    endif()
    list(APPEND RAYD_CUDA_GENCODE_FLAGS_LIST
        "-gencode=arch=compute_${RAYD_CUDA_ARCH},code=sm_${RAYD_CUDA_ARCH}")
endforeach()
if(NOT RAYD_CUDA_PTX_ARCH STREQUAL "")
    if(NOT RAYD_CUDA_PTX_ARCH MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Invalid CUDA architecture '${RAYD_CUDA_PTX_ARCH}' in RAYD_CUDA_PTX_ARCH.")
    endif()
    list(APPEND RAYD_CUDA_GENCODE_FLAGS_LIST
        "-gencode=arch=compute_${RAYD_CUDA_PTX_ARCH},code=compute_${RAYD_CUDA_PTX_ARCH}")
endif()
string(JOIN " " RAYD_CUDA_GENCODE_FLAGS ${RAYD_CUDA_GENCODE_FLAGS_LIST})

set(RAYD_INCLUDE_DIR include/rayd)
set(RAYD_SOURCE_DIR src)
set(RAYD_CORE_SOURCE_FILES
    ${RAYD_INCLUDE_DIR}/rayd.h
    ${RAYD_INCLUDE_DIR}/types.h
    ${RAYD_INCLUDE_DIR}/utils.h

    ${RAYD_INCLUDE_DIR}/ray.h
    ${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation.h
    ${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_params.h
    ${RAYD_INCLUDE_DIR}/multipath/diffraction_paths.h
    ${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_params.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_epc.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation.h
    ${RAYD_INCLUDE_DIR}/multipath/segment_visibility.h
    ${RAYD_INCLUDE_DIR}/transform.h

    ${RAYD_INCLUDE_DIR}/edge/edge.h

    ${RAYD_INCLUDE_DIR}/mesh.h
    ${RAYD_SOURCE_DIR}/mesh.cpp

    ${RAYD_INCLUDE_DIR}/optix.h

    ${RAYD_INCLUDE_DIR}/surfel/surfel_optix.h
    ${RAYD_INCLUDE_DIR}/surfel/surfel_trace_params.h
    ${RAYD_SOURCE_DIR}/surfel/surfel_optix.cpp
    ${RAYD_INCLUDE_DIR}/surfel/surfel.h
    ${RAYD_SOURCE_DIR}/surfel/surfel.cpp

    ${RAYD_INCLUDE_DIR}/scene/scene_optix.h
    ${RAYD_SOURCE_DIR}/scene/scene_optix.cpp
    ${RAYD_INCLUDE_DIR}/trace/trace_backend.h
    ${RAYD_INCLUDE_DIR}/trace/optix_trace_backend.h
    ${RAYD_SOURCE_DIR}/trace/optix_trace_backend.cpp
    ${RAYD_INCLUDE_DIR}/trace/cuda_trace_backend.h
    ${RAYD_INCLUDE_DIR}/trace/triangle_bvh_gpu.h
    ${RAYD_SOURCE_DIR}/trace/cuda_trace_backend.cpp
    ${RAYD_INCLUDE_DIR}/multipath/pipelines.h
    ${RAYD_SOURCE_DIR}/multipath/pipelines.cpp
    ${RAYD_INCLUDE_DIR}/multipath/reflection_trace_params.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_epc_params.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_epc_field.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_params.h
    ${RAYD_INCLUDE_DIR}/multipath/segment_visibility_params.h
    ${RAYD_INCLUDE_DIR}/multipath/reflection_dedup.h
    ${RAYD_INCLUDE_DIR}/edge/scene_edge.h
    ${RAYD_INCLUDE_DIR}/edge/scene_edge_optix.h
    ${RAYD_SOURCE_DIR}/edge/scene_edge.cpp
    ${RAYD_SOURCE_DIR}/edge/scene_edge_optix.cpp
    ${RAYD_INCLUDE_DIR}/edge/edge_bvh.h
    ${RAYD_INCLUDE_DIR}/edge/edge_bvh_config.h
    ${RAYD_INCLUDE_DIR}/edge/edge_optix_params.h

    ${RAYD_INCLUDE_DIR}/scene/scene.h
    ${RAYD_SOURCE_DIR}/scene/scene_internal.h
    ${RAYD_SOURCE_DIR}/scene/scene.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_intersect.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_internal.h
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_config.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_reflection.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_reflection_accum.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_epc.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_diffraction_paths.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_diffraction_accum.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_diffraction_coherent.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_multipath_visibility.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_custom_op.cpp

    ${RAYD_SOURCE_DIR}/optix.cpp
    ${RAYD_INCLUDE_DIR}/native_launch_audit.h
    ${RAYD_SOURCE_DIR}/native_launch_audit.cpp

)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rayd_cuda.cmake")

if(WIN32)
    # nvcc drives cl.exe, so the generated .bat wrappers need an initialized MSVC
    # environment. A developer shell already provides one; the Visual Studio
    # generator instead exposes its instance path so each wrapper can set one up.
    if(DEFINED ENV{VSCMD_VER})
        set(RAYD_MSVC_SETUP "")
    elseif(CMAKE_GENERATOR_INSTANCE)
        set(RAYD_VSDEVCMD "${CMAKE_GENERATOR_INSTANCE}/Common7/Tools/VsDevCmd.bat")
        set(RAYD_MSVC_SETUP "call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n")
    else()
        message(FATAL_ERROR
            "RayD's Windows CUDA build needs an initialized MSVC environment. "
            "Run scripts/build_local.cmd or configure from a Visual Studio developer shell.")
    endif()
endif()

# nvcc numeric flags are a contract. shared/contracts/compile_policy.json and
# docs/adr/0035-cuda-compile-flag-policy.md declare the numeric profile of every
# unit below, and tests/test_compile_flag_policy_contract.py re-derives it from
# this file. EXTRA_FLAGS is the only place a numeric flag may be introduced here;
# the shared command shape in cmake/rayd_cuda.cmake carries none for objects.

# Edge BVH build/refit/query, split between the backend and the shared core.
# ADR-0035 profile: nvcc_default for all six.
rayd_cuda_object(
    NAME edge_bvh
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_bvh.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_bvh.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/bvh_build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/bvh_types.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/refit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/topology.h"
)
rayd_cuda_object(
    NAME shared_bvh_core_build
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/bvh/build.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/refit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/topology.h"
)
rayd_cuda_object(
    NAME shared_bvh_build
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/edge/bvh_build.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/bvh_build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/refit.h"
)
rayd_cuda_object(
    NAME shared_bvh_query
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/edge/bvh_query.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/bvh_query.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/edge_distance_math.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/traversal_common.cuh"
)
rayd_cuda_object(
    NAME shared_edge_distance
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/edge/edge_distance.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/edge_distance.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/edge_distance_math.h"
)
rayd_cuda_object(
    NAME shared_edge_aabb
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/edge/edge_aabb.cu"
    POSIX_NO_EXTENDED_LAMBDA
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/edge/edge_aabb.h"
)

# Pure-CUDA triangle BVH: the primitive-specific traversal kernels live in the
# shared tree; the build/refit/query orchestrator lives in the drjit backend.
rayd_cuda_object(
    NAME shared_bvh_triangle_query
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/bvh/triangle_query.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_query.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_query_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/cuda_bvh_traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_intersect.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/traversal_common.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/topology.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
)
rayd_cuda_object(
    NAME triangle_bvh
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/trace/triangle_bvh.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/trace/triangle_bvh_gpu.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/build.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_query.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/topology.h"
)

# CUDA fused multipath executor (P4 Stage D): runs the traverser-templated
# multipath algorithm bodies with CudaBvhTraverser over the scene triangle BVH.
# Object-compiled (host-callable launchers) like triangle_bvh.cu; needs no OptiX
# headers because the migrated params/algo headers are host-safe. It is the only
# object translation unit compiled with --use_fast_math.
#
# ADR-0035 profile: fast_math. Frozen divergence D1: the Torch backend compiles
# its counterpart, src/torch_ext/scene/multipath_cuda.cu, at nvcc defaults over
# the same shared multipath algorithm headers. Do not align either side here.
rayd_cuda_object(
    NAME cuda_multipath
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/trace/cuda_multipath.cu"
    EXTRA_FLAGS --use_fast_math
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/trace/cuda_multipath_gpu.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/cuda_bvh_traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_query_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_query.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/triangle_intersect.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/traversal_common.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/bvh/topology.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_trace_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/segment_visibility_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_accumulation_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_epc_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/diffraction_accumulation_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/diffraction_contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_trace_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/segment_visibility_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_epc_params.h"
)

rayd_cuda_object(
    NAME reflection_dedup
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_dedup.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_dedup.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/multipath/reflection_dedup.h"
)
rayd_cuda_object(
    NAME shared_reflection_dedup
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/src/multipath/reflection_dedup.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/multipath/reflection_dedup.h"
)

rayd_cuda_object(
    NAME reflection_epc_field
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_epc_field.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_field.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/field_math.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/multipath/reflection_epc_field_device.cuh"
)

rayd_cuda_object(
    NAME diffraction_accumulation_ad
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_accumulation_ad.cu"
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_ad.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/multipath/diffraction_accumulation_ad_device.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/multipath/diffraction_accumulation_ad_vjp_device.cuh"
)

# OptiX device programs. Wheels consume the committed *_ptx.h headers; the
# RAYD_REGENERATE_*_PTX options rebuild them from source against an OptiX SDK.
#
# ADR-0035 profile: fast_math at compute_70 for all eight, set once in the
# shared PTX command shape. Divergence D2: the Torch backend targets compute_75.
# Divergence D9: every option below defaults OFF, so these flags describe how a
# module is regenerated, not provably how the committed header was produced.
rayd_embed_ptx(
    NAME reflection_trace
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_trace.cu"
    HEADER multipath/reflection_trace_ptx.h
    OPTION RAYD_REGENERATE_REFLECTION_TRACE_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_trace_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/math/vec3.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/reflection/reflection_geometry.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/device_hit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/optix_traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_trace_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_trace_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_trace_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
)
rayd_embed_ptx(
    NAME reflection_epc
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_epc.cu"
    HEADER multipath/reflection_epc_ptx.h
    OPTION RAYD_REGENERATE_REFLECTION_EPC_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/device_hit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_epc_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/reflection_epc_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_epc_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/reflection/reflection_geometry.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/reflection/epc_chain.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
)
rayd_embed_ptx(
    NAME reflection_accumulation
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_accumulation.cu"
    HEADER multipath/reflection_accumulation_ptx.h
    OPTION RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/field_math.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_accumulation_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/reflection_accumulation_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/optix_traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/device_hit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
)
rayd_embed_ptx(
    NAME diffraction_accumulation
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_accumulation.cu"
    HEADER multipath/diffraction_accumulation_ptx.h
    OPTION RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/diffraction_contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/diffraction_accumulation_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/diffraction_accumulation_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/rayd.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/utils.h"
)
rayd_embed_ptx(
    NAME diffraction_paths
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_paths.cu"
    HEADER multipath/diffraction_paths_ptx.h
    OPTION RAYD_REGENERATE_DIFFRACTION_PATHS_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/diffraction_contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/diffraction_paths_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/diffraction_paths_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/rayd.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/utils.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation.h"
)
rayd_embed_ptx(
    NAME segment_visibility
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/segment_visibility.cu"
    HEADER multipath/segment_visibility_ptx.h
    OPTION RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/segment_visibility_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/device_hit.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/segment_visibility_params.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/segment_visibility_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/multipath/segment_visibility_algo.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/traverser.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
)
rayd_embed_ptx(
    NAME surfel_trace
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/surfel/surfel_trace.cu"
    HEADER surfel/surfel_trace_ptx.h
    OPTION RAYD_REGENERATE_SURFEL_TRACE_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/surfel/surfel_trace_params.h"
)
rayd_embed_ptx(
    NAME edge_optix
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_optix.cu"
    HEADER edge/edge_optix_ptx.h
    OPTION RAYD_REGENERATE_EDGE_OPTIX_PTX
    OUT_SOURCES RAYD_CORE_SOURCE_FILES
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_optix_params.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/edge/edge_distance_math.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include/rayd/shared/math/vec3.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/scene_edge_contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/scene_edge_device.cuh"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/optix/pipeline_contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/contracts.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/numeric_policy.h"
        "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/rt/qualifiers.h"
)

set(RAYD_NANOBIND_ARGS NB_DOMAIN drjit)
if(RAYD_NANOBIND_NOMINSIZE)
    list(APPEND RAYD_NANOBIND_ARGS NOMINSIZE)
endif()
if(RAYD_ENABLE_LTO)
    list(APPEND RAYD_NANOBIND_ARGS LTO)
endif()

add_library(rayd_core STATIC ${RAYD_CORE_SOURCE_FILES})
if(RAYD_REGENERATE_REFLECTION_TRACE_PTX OR
   RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX OR
   RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX OR
   RAYD_REGENERATE_DIFFRACTION_PATHS_PTX OR
   RAYD_REGENERATE_REFLECTION_EPC_PTX OR
   RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX OR
   RAYD_REGENERATE_SURFEL_TRACE_PTX OR
   RAYD_REGENERATE_EDGE_OPTIX_PTX)
    target_include_directories(rayd_core BEFORE PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()
target_include_directories(rayd_core
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${RAYD_SHARED_INCLUDE_DIR}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${NB_DIR}/include
)
target_compile_definitions(rayd_core PRIVATE
    RAYD_OPTIX_MODULE_OPT_LEVEL=${RAYD_OPTIX_MODULE_OPT_LEVEL}
    RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL=${RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL}
    RAYD_OPTIX_EXCEPTION_FLAGS=${RAYD_OPTIX_EXCEPTION_FLAGS}
    RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS=${RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS}
)

if(WIN32)
    target_compile_options(rayd_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O2>
    )
endif()

set(PYRAYD_LIBRARIES drjit drjit-core drjit-extra nanothread)
if(WIN32)
    # version: OptiX driver-module version query.
    # cfgmgr32/advapi32: driver-store lookup for nvoptix.dll in optix_available().
    list(APPEND PYRAYD_LIBRARIES version cfgmgr32 advapi32)
endif()
target_link_libraries(rayd_core PUBLIC ${PYRAYD_LIBRARIES} ${CUDA_LIBRARIES})
set_property(TARGET rayd_core PROPERTY CXX_STANDARD 17)
set_target_properties(rayd_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

nanobind_add_module(_C ${RAYD_NANOBIND_ARGS} ${RAYD_SOURCE_DIR}/rayd.cpp)
target_link_libraries(_C PRIVATE rayd_core)
if(WIN32)
    target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O2>
    )
endif()

install(TARGETS _C DESTINATION rayd/drjit)
install(TARGETS rayd_core
    ARCHIVE DESTINATION rayd/drjit/lib
    LIBRARY DESTINATION rayd/drjit/lib
    RUNTIME DESTINATION rayd/drjit/bin
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rayd
    DESTINATION rayd/drjit/include
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)
install(DIRECTORY ${RAYD_SHARED_INCLUDE_DIR}/rayd/shared
    DESTINATION rayd/drjit/include/rayd
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)

set_target_properties(_C PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_property(TARGET _C PROPERTY CXX_STANDARD 17)
set_target_properties(_C PROPERTIES PREFIX "")

