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

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
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
    ${RAYD_SOURCE_DIR}/scene/scene_multipath.cpp
    ${RAYD_SOURCE_DIR}/scene/scene_custom_op.cpp

    ${RAYD_SOURCE_DIR}/optix.cpp
    ${RAYD_INCLUDE_DIR}/native_launch_audit.h
    ${RAYD_SOURCE_DIR}/native_launch_audit.cpp

)
if(WIN32)
    set(RAYD_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.obj")
    set(RAYD_VSDEVCMD "${CMAKE_GENERATOR_INSTANCE}/Common7/Tools/VsDevCmd.bat")
    set(RAYD_CUDA_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_edge_bvh.bat")
    file(GENERATE OUTPUT "${RAYD_CUDA_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_bvh.cu\" ${RAYD_CUDA_GENCODE_FLAGS} ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_CUDA_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_CUDA_OBJECT}"
        COMMAND "${RAYD_CUDA_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_CUDA_BUILD_SCRIPT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_bvh.h"
        VERBATIM
    )
else()
    set(RAYD_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.o")
    add_custom_command(
        OUTPUT "${RAYD_CUDA_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" -std=c++17 -Xcompiler=-fPIC -c
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_bvh.cu"
                ${RAYD_CUDA_GENCODE_FLAGS_LIST}
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_CUDA_OBJECT}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_bvh.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYD_CUDA_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_CUDA_OBJECT}")

set(RAYD_REFLECTION_DEDUP_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_dedup.cu")
if(WIN32)
    set(RAYD_REFLECTION_DEDUP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/reflection_dedup.obj")
    set(RAYD_REFLECTION_DEDUP_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_reflection_dedup.bat")
    file(GENERATE OUTPUT "${RAYD_REFLECTION_DEDUP_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${RAYD_REFLECTION_DEDUP_CU}\" ${RAYD_CUDA_GENCODE_FLAGS} ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_REFLECTION_DEDUP_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_DEDUP_OBJECT}"
        COMMAND "${RAYD_REFLECTION_DEDUP_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_REFLECTION_DEDUP_BUILD_SCRIPT}"
            "${RAYD_REFLECTION_DEDUP_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_dedup.h"
        VERBATIM
    )
else()
    set(RAYD_REFLECTION_DEDUP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/reflection_dedup.o")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_DEDUP_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" --extended-lambda -std=c++17 -Xcompiler=-fPIC -c
                "${RAYD_REFLECTION_DEDUP_CU}"
                ${RAYD_CUDA_GENCODE_FLAGS_LIST}
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_REFLECTION_DEDUP_OBJECT}"
        DEPENDS
            "${RAYD_REFLECTION_DEDUP_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_dedup.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYD_REFLECTION_DEDUP_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_DEDUP_OBJECT}")

set(RAYD_REFLECTION_EPC_FIELD_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_epc_field.cu")
if(WIN32)
    set(RAYD_REFLECTION_EPC_FIELD_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_field.obj")
    set(RAYD_REFLECTION_EPC_FIELD_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_reflection_epc_field.bat")
    file(GENERATE OUTPUT "${RAYD_REFLECTION_EPC_FIELD_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${RAYD_REFLECTION_EPC_FIELD_CU}\" ${RAYD_CUDA_GENCODE_FLAGS} ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_REFLECTION_EPC_FIELD_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_EPC_FIELD_OBJECT}"
        COMMAND "${RAYD_REFLECTION_EPC_FIELD_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_REFLECTION_EPC_FIELD_BUILD_SCRIPT}"
            "${RAYD_REFLECTION_EPC_FIELD_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_field.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        VERBATIM
    )
else()
    set(RAYD_REFLECTION_EPC_FIELD_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc_field.o")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_EPC_FIELD_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" --extended-lambda -std=c++17 -Xcompiler=-fPIC -c
                "${RAYD_REFLECTION_EPC_FIELD_CU}"
                ${RAYD_CUDA_GENCODE_FLAGS_LIST}
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_REFLECTION_EPC_FIELD_OBJECT}"
        DEPENDS
            "${RAYD_REFLECTION_EPC_FIELD_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_field.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYD_REFLECTION_EPC_FIELD_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_EPC_FIELD_OBJECT}")

set(RAYD_DIFFRACTION_ACCUMULATION_AD_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_accumulation_ad.cu")
if(WIN32)
    set(RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_ad.obj")
    set(RAYD_DIFFRACTION_ACCUMULATION_AD_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_diffraction_accumulation_ad.bat")
    file(GENERATE OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_AD_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${RAYD_DIFFRACTION_ACCUMULATION_AD_CU}\" ${RAYD_CUDA_GENCODE_FLAGS} ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}"
        COMMAND "${RAYD_DIFFRACTION_ACCUMULATION_AD_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_DIFFRACTION_ACCUMULATION_AD_BUILD_SCRIPT}"
            "${RAYD_DIFFRACTION_ACCUMULATION_AD_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_ad.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        VERBATIM
    )
else()
    set(RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation_ad.o")
    add_custom_command(
        OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" --extended-lambda -std=c++17 -Xcompiler=-fPIC -c
                "${RAYD_DIFFRACTION_ACCUMULATION_AD_CU}"
                ${RAYD_CUDA_GENCODE_FLAGS_LIST}
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}"
        DEPENDS
            "${RAYD_DIFFRACTION_ACCUMULATION_AD_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_ad.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/native_launch_audit.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_DIFFRACTION_ACCUMULATION_AD_OBJECT}")

set(RAYD_REFLECTION_TRACE_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_trace.cu")
set(RAYD_REFLECTION_TRACE_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/reflection_trace_ptx.h")
set(RAYD_REFLECTION_TRACE_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_trace_ptx.h")

if(RAYD_REGENERATE_REFLECTION_TRACE_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate reflection_trace PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_REFLECTION_TRACE_PTX.")
    endif()

    set(RAYD_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace.ptx")
    if(WIN32)
        set(RAYD_REFLECTION_TRACE_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_reflection_trace_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_REFLECTION_TRACE_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_REFLECTION_TRACE_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_REFLECTION_TRACE_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_TRACE_PTX}"
            COMMAND "${RAYD_REFLECTION_TRACE_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_REFLECTION_TRACE_BUILD_SCRIPT}"
                "${RAYD_REFLECTION_TRACE_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_trace_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_TRACE_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_REFLECTION_TRACE_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_REFLECTION_TRACE_PTX}"
            DEPENDS
                "${RAYD_REFLECTION_TRACE_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_trace_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_TRACE_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_REFLECTION_TRACE_PTX}
            -DOUTPUT_FILE=${RAYD_REFLECTION_TRACE_PTX_H}
            -DVAR_NAME=reflection_trace_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_REFLECTION_TRACE_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_REFLECTION_TRACE_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_TRACE_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_REFLECTION_TRACE_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed reflection_trace PTX header at "
            "${RAYD_REFLECTION_TRACE_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_REFLECTION_TRACE_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_TRACE_PTX_H_COMMITTED}")
endif()

set(RAYD_REFLECTION_EPC_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_epc.cu")
set(RAYD_REFLECTION_EPC_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/reflection_epc_ptx.h")
set(RAYD_REFLECTION_EPC_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_ptx.h")

if(RAYD_REGENERATE_REFLECTION_EPC_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate reflection_epc PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_REFLECTION_EPC_PTX.")
    endif()

    set(RAYD_REFLECTION_EPC_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_epc.ptx")
    if(WIN32)
        set(RAYD_REFLECTION_EPC_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_reflection_epc_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_REFLECTION_EPC_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_REFLECTION_EPC_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_REFLECTION_EPC_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_EPC_PTX}"
            COMMAND "${RAYD_REFLECTION_EPC_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_REFLECTION_EPC_BUILD_SCRIPT}"
                "${RAYD_REFLECTION_EPC_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_EPC_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_REFLECTION_EPC_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_REFLECTION_EPC_PTX}"
            DEPENDS
                "${RAYD_REFLECTION_EPC_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_epc_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_EPC_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_REFLECTION_EPC_PTX}
            -DOUTPUT_FILE=${RAYD_REFLECTION_EPC_PTX_H}
            -DVAR_NAME=reflection_epc_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_REFLECTION_EPC_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_REFLECTION_EPC_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_EPC_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_REFLECTION_EPC_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed reflection_epc PTX header at "
            "${RAYD_REFLECTION_EPC_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_REFLECTION_EPC_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_EPC_PTX_H_COMMITTED}")
endif()

set(RAYD_REFLECTION_ACCUMULATION_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_accumulation.cu")
set(RAYD_REFLECTION_ACCUMULATION_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/reflection_accumulation_ptx.h")
set(RAYD_REFLECTION_ACCUMULATION_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_ptx.h")

if(RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate reflection_accumulation PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX.")
    endif()

    set(RAYD_REFLECTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_accumulation.ptx")
    if(WIN32)
        set(RAYD_REFLECTION_ACCUMULATION_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_reflection_accumulation_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_REFLECTION_ACCUMULATION_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_REFLECTION_ACCUMULATION_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_REFLECTION_ACCUMULATION_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_ACCUMULATION_PTX}"
            COMMAND "${RAYD_REFLECTION_ACCUMULATION_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_REFLECTION_ACCUMULATION_BUILD_SCRIPT}"
                "${RAYD_REFLECTION_ACCUMULATION_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_REFLECTION_ACCUMULATION_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_REFLECTION_ACCUMULATION_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_REFLECTION_ACCUMULATION_PTX}"
            DEPENDS
                "${RAYD_REFLECTION_ACCUMULATION_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/reflection_accumulation_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_ACCUMULATION_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_REFLECTION_ACCUMULATION_PTX}
            -DOUTPUT_FILE=${RAYD_REFLECTION_ACCUMULATION_PTX_H}
            -DVAR_NAME=reflection_accumulation_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_REFLECTION_ACCUMULATION_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_REFLECTION_ACCUMULATION_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_ACCUMULATION_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_REFLECTION_ACCUMULATION_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed reflection_accumulation PTX header at "
            "${RAYD_REFLECTION_ACCUMULATION_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_REFLECTION_ACCUMULATION_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_ACCUMULATION_PTX_H_COMMITTED}")
endif()

set(RAYD_DIFFRACTION_ACCUMULATION_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_accumulation.cu")
set(RAYD_DIFFRACTION_ACCUMULATION_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/diffraction_accumulation_ptx.h")
set(RAYD_DIFFRACTION_ACCUMULATION_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_ptx.h")

if(RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate diffraction_accumulation PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX.")
    endif()

    set(RAYD_DIFFRACTION_ACCUMULATION_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_accumulation.ptx")
    if(WIN32)
        set(RAYD_DIFFRACTION_ACCUMULATION_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_diffraction_accumulation_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_DIFFRACTION_ACCUMULATION_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_DIFFRACTION_ACCUMULATION_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_PTX}"
            COMMAND "${RAYD_DIFFRACTION_ACCUMULATION_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_DIFFRACTION_ACCUMULATION_BUILD_SCRIPT}"
                "${RAYD_DIFFRACTION_ACCUMULATION_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_params.h"
                "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
                "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_DIFFRACTION_ACCUMULATION_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_DIFFRACTION_ACCUMULATION_PTX}"
            DEPENDS
                "${RAYD_DIFFRACTION_ACCUMULATION_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_accumulation_params.h"
                "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_types.h"
                "${RAYD_SHARED_INCLUDE_DIR}/rayd/shared/utd/utd_math.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_DIFFRACTION_ACCUMULATION_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_DIFFRACTION_ACCUMULATION_PTX}
            -DOUTPUT_FILE=${RAYD_DIFFRACTION_ACCUMULATION_PTX_H}
            -DVAR_NAME=diffraction_accumulation_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_DIFFRACTION_ACCUMULATION_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_DIFFRACTION_ACCUMULATION_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_DIFFRACTION_ACCUMULATION_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_DIFFRACTION_ACCUMULATION_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed diffraction_accumulation PTX header at "
            "${RAYD_DIFFRACTION_ACCUMULATION_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_DIFFRACTION_ACCUMULATION_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_DIFFRACTION_ACCUMULATION_PTX_H_COMMITTED}")
endif()

set(RAYD_DIFFRACTION_PATHS_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/diffraction_paths.cu")
set(RAYD_DIFFRACTION_PATHS_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/diffraction_paths_ptx.h")
set(RAYD_DIFFRACTION_PATHS_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_ptx.h")

if(RAYD_REGENERATE_DIFFRACTION_PATHS_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate diffraction_paths PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_DIFFRACTION_PATHS_PTX.")
    endif()

    set(RAYD_DIFFRACTION_PATHS_PTX "${CMAKE_CURRENT_BINARY_DIR}/diffraction_paths.ptx")
    if(WIN32)
        set(RAYD_DIFFRACTION_PATHS_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_diffraction_paths_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_DIFFRACTION_PATHS_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_DIFFRACTION_PATHS_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_DIFFRACTION_PATHS_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_DIFFRACTION_PATHS_PTX}"
            COMMAND "${RAYD_DIFFRACTION_PATHS_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_DIFFRACTION_PATHS_BUILD_SCRIPT}"
                "${RAYD_DIFFRACTION_PATHS_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_DIFFRACTION_PATHS_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_DIFFRACTION_PATHS_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_DIFFRACTION_PATHS_PTX}"
            DEPENDS
                "${RAYD_DIFFRACTION_PATHS_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/diffraction_paths_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_DIFFRACTION_PATHS_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_DIFFRACTION_PATHS_PTX}
            -DOUTPUT_FILE=${RAYD_DIFFRACTION_PATHS_PTX_H}
            -DVAR_NAME=diffraction_paths_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_DIFFRACTION_PATHS_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_DIFFRACTION_PATHS_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_DIFFRACTION_PATHS_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_DIFFRACTION_PATHS_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed diffraction_paths PTX header at "
            "${RAYD_DIFFRACTION_PATHS_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_DIFFRACTION_PATHS_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_DIFFRACTION_PATHS_PTX_H_COMMITTED}")
endif()

set(RAYD_SEGMENT_VISIBILITY_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/segment_visibility.cu")
set(RAYD_SEGMENT_VISIBILITY_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/multipath/segment_visibility_ptx.h")
set(RAYD_SEGMENT_VISIBILITY_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/segment_visibility_ptx.h")

if(RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate segment_visibility PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX.")
    endif()

    set(RAYD_SEGMENT_VISIBILITY_PTX "${CMAKE_CURRENT_BINARY_DIR}/segment_visibility.ptx")
    if(WIN32)
        set(RAYD_SEGMENT_VISIBILITY_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_segment_visibility_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_SEGMENT_VISIBILITY_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_SEGMENT_VISIBILITY_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_SEGMENT_VISIBILITY_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_SEGMENT_VISIBILITY_PTX}"
            COMMAND "${RAYD_SEGMENT_VISIBILITY_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_SEGMENT_VISIBILITY_BUILD_SCRIPT}"
                "${RAYD_SEGMENT_VISIBILITY_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/segment_visibility_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_SEGMENT_VISIBILITY_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_SEGMENT_VISIBILITY_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_SEGMENT_VISIBILITY_PTX}"
            DEPENDS
                "${RAYD_SEGMENT_VISIBILITY_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/multipath/segment_visibility_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_SEGMENT_VISIBILITY_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_SEGMENT_VISIBILITY_PTX}
            -DOUTPUT_FILE=${RAYD_SEGMENT_VISIBILITY_PTX_H}
            -DVAR_NAME=segment_visibility_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_SEGMENT_VISIBILITY_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_SEGMENT_VISIBILITY_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_SEGMENT_VISIBILITY_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_SEGMENT_VISIBILITY_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed segment_visibility PTX header at "
            "${RAYD_SEGMENT_VISIBILITY_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_SEGMENT_VISIBILITY_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_SEGMENT_VISIBILITY_PTX_H_COMMITTED}")
endif()

set(RAYD_SURFEL_TRACE_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/surfel/surfel_trace.cu")
set(RAYD_SURFEL_TRACE_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/surfel/surfel_trace_ptx.h")
set(RAYD_SURFEL_TRACE_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/surfel/surfel_trace_ptx.h")

if(RAYD_REGENERATE_SURFEL_TRACE_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate surfel_trace PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_SURFEL_TRACE_PTX.")
    endif()

    set(RAYD_SURFEL_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/surfel_trace.ptx")
    if(WIN32)
        set(RAYD_SURFEL_TRACE_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_surfel_trace_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_SURFEL_TRACE_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_SURFEL_TRACE_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_SURFEL_TRACE_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_SURFEL_TRACE_PTX}"
            COMMAND "${RAYD_SURFEL_TRACE_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_SURFEL_TRACE_BUILD_SCRIPT}"
                "${RAYD_SURFEL_TRACE_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/surfel/surfel_trace_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_SURFEL_TRACE_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_SURFEL_TRACE_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_SURFEL_TRACE_PTX}"
            DEPENDS
                "${RAYD_SURFEL_TRACE_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/surfel/surfel_trace_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_SURFEL_TRACE_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_SURFEL_TRACE_PTX}
            -DOUTPUT_FILE=${RAYD_SURFEL_TRACE_PTX_H}
            -DVAR_NAME=surfel_trace_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_SURFEL_TRACE_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_SURFEL_TRACE_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_SURFEL_TRACE_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_SURFEL_TRACE_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed surfel_trace PTX header at "
            "${RAYD_SURFEL_TRACE_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_SURFEL_TRACE_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_SURFEL_TRACE_PTX_H_COMMITTED}")
endif()

set(RAYD_EDGE_OPTIX_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/edge/edge_optix.cu")
set(RAYD_EDGE_OPTIX_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/rayd/edge/edge_optix_ptx.h")
set(RAYD_EDGE_OPTIX_PTX_H_COMMITTED
    "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_optix_ptx.h")

if(RAYD_REGENERATE_EDGE_OPTIX_PTX)
    find_path(OPTIX_INCLUDE_DIR optix.h
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
        message(FATAL_ERROR
            "Could not locate OptiX SDK headers needed to regenerate edge_optix PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable RAYD_REGENERATE_EDGE_OPTIX_PTX.")
    endif()

    set(RAYD_EDGE_OPTIX_PTX "${CMAKE_CURRENT_BINARY_DIR}/edge_optix.ptx")
    if(WIN32)
        set(RAYD_EDGE_OPTIX_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_edge_optix_ptx.bat")
        file(GENERATE OUTPUT "${RAYD_EDGE_OPTIX_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${RAYD_EDGE_OPTIX_CU}\" -I\"${OPTIX_INCLUDE_DIR}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${RAYD_EDGE_OPTIX_PTX}\"\r\n\
")
        add_custom_command(
            OUTPUT "${RAYD_EDGE_OPTIX_PTX}"
            COMMAND "${RAYD_EDGE_OPTIX_BUILD_SCRIPT}"
            DEPENDS
                "${RAYD_EDGE_OPTIX_BUILD_SCRIPT}"
                "${RAYD_EDGE_OPTIX_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_optix_params.h"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${RAYD_EDGE_OPTIX_PTX}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${RAYD_EDGE_OPTIX_CU}"
                    -I"${OPTIX_INCLUDE_DIR}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${RAYD_EDGE_OPTIX_PTX}"
            DEPENDS
                "${RAYD_EDGE_OPTIX_CU}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/edge/edge_optix_params.h"
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${RAYD_EDGE_OPTIX_PTX_H}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${RAYD_EDGE_OPTIX_PTX}
            -DOUTPUT_FILE=${RAYD_EDGE_OPTIX_PTX_H}
            -DVAR_NAME=edge_optix_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${RAYD_EDGE_OPTIX_PTX}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${RAYD_EDGE_OPTIX_PTX_H}" PROPERTIES GENERATED TRUE)
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_EDGE_OPTIX_PTX_H}")
else()
    if(NOT EXISTS "${RAYD_EDGE_OPTIX_PTX_H_COMMITTED}")
        message(FATAL_ERROR
            "Missing committed edge_optix PTX header at "
            "${RAYD_EDGE_OPTIX_PTX_H_COMMITTED}. "
            "Restore it or enable RAYD_REGENERATE_EDGE_OPTIX_PTX with OptiX SDK headers available.")
    endif()
    list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_EDGE_OPTIX_PTX_H_COMMITTED}")
endif()

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
    list(APPEND PYRAYD_LIBRARIES version)
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
)

set_target_properties(_C PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_property(TARGET _C PROPERTY CXX_STANDARD 17)
set_target_properties(_C PROPERTIES PREFIX "")

