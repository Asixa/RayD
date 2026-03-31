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

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib, sys; spec = importlib.util.find_spec('drjit'); sys.stdout.write(str(pathlib.Path(next(iter(spec.submodule_search_locations))).resolve() / 'cmake')) if spec and spec.submodule_search_locations else sys.exit(1)"
    RESULT_VARIABLE DRJIT_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE DRJIT_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT DRJIT_CMAKE_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed Dr.Jit CMake package. Ensure drjit>=1.2.0 is installed in the build environment.")
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

find_package(CUDA 11.0 REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

find_path(OPTIX_INCLUDE_DIR optix.h
    HINTS
        ENV OPTIX_PATH
        "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 9.1.0/include"
        "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.1.0/include"
        "$ENV{PROGRAMDATA}/NVIDIA Corporation/OptiX SDK 8.0.0/include"
)
if(NOT OPTIX_INCLUDE_DIR)
    message(FATAL_ERROR "Could not locate OptiX SDK headers. Set OPTIX_INCLUDE_DIR or OPTIX_PATH.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include_directories(
    include/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set(RAYD_CUDA_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
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

set(RAYD_INCLUDE_DIR include/rayd)
set(RAYD_SOURCE_DIR src)
set(RAYD_CORE_SOURCE_FILES
    ${RAYD_INCLUDE_DIR}/rayd.h
    ${RAYD_INCLUDE_DIR}/constants.h
    ${RAYD_INCLUDE_DIR}/fwd.h
    ${RAYD_INCLUDE_DIR}/types.h
    ${RAYD_INCLUDE_DIR}/utils.h

    ${RAYD_INCLUDE_DIR}/ray.h
    ${RAYD_INCLUDE_DIR}/intersection.h
    ${RAYD_INCLUDE_DIR}/reflection.h
    ${RAYD_INCLUDE_DIR}/transform.h

    ${RAYD_INCLUDE_DIR}/edge.h

    ${RAYD_INCLUDE_DIR}/mesh.h
    ${RAYD_SOURCE_DIR}/mesh.cpp

    ${RAYD_INCLUDE_DIR}/optix.h

    ${RAYD_INCLUDE_DIR}/camera.h
    ${RAYD_SOURCE_DIR}/camera.cpp

    ${RAYD_INCLUDE_DIR}/scene/scene_optix.h
    ${RAYD_SOURCE_DIR}/scene/scene_optix.cpp
    ${RAYD_SOURCE_DIR}/multipath/reflection_trace_host.h
    ${RAYD_SOURCE_DIR}/multipath/reflection_trace_params.h
    ${RAYD_SOURCE_DIR}/multipath/reflection_trace_host.cpp
    ${RAYD_SOURCE_DIR}/multipath/reflection_dedup.h
    ${RAYD_INCLUDE_DIR}/scene/scene_edge.h
    ${RAYD_SOURCE_DIR}/scene/scene_edge.cpp
    ${RAYD_SOURCE_DIR}/scene/edge_bvh.h
    ${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc.h
    ${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_build.inc
    ${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_kernels.inc
    ${RAYD_SOURCE_DIR}/scene/experimental/scene_edge_ploc.inc

    ${RAYD_INCLUDE_DIR}/scene/scene.h
    ${RAYD_SOURCE_DIR}/scene/scene.cpp

    ${RAYD_SOURCE_DIR}/optix.cpp

    ${RAYD_INCLUDE_DIR}/slang/interop.h
    ${RAYD_INCLUDE_DIR}/slang/interop_types.h
    ${RAYD_SOURCE_DIR}/slang_interop.cpp
)
if(WIN32)
    set(RAYD_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.obj")
    set(RAYD_VSDEVCMD "${CMAKE_GENERATOR_INSTANCE}/Common7/Tools/VsDevCmd.bat")
    set(RAYD_CUDA_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_edge_bvh.bat")
    file(GENERATE OUTPUT "${RAYD_CUDA_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYD_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.cu\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_CUDA_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_CUDA_OBJECT}"
        COMMAND "${RAYD_CUDA_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_CUDA_BUILD_SCRIPT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_build.inc"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_kernels.inc"
        VERBATIM
    )
else()
    set(RAYD_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.o")
    add_custom_command(
        OUTPUT "${RAYD_CUDA_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" -std=c++17 -Xcompiler=-fPIC -c
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.cu"
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_CUDA_OBJECT}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/edge_bvh.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_build.inc"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/scene/experimental/edge_bvh_ploc_kernels.inc"
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
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${RAYD_REFLECTION_DEDUP_CU}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYD_REFLECTION_DEDUP_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_DEDUP_OBJECT}"
        COMMAND "${RAYD_REFLECTION_DEDUP_BUILD_SCRIPT}"
        DEPENDS
            "${RAYD_REFLECTION_DEDUP_BUILD_SCRIPT}"
            "${RAYD_REFLECTION_DEDUP_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_dedup.h"
        VERBATIM
    )
else()
    set(RAYD_REFLECTION_DEDUP_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/reflection_dedup.o")
    add_custom_command(
        OUTPUT "${RAYD_REFLECTION_DEDUP_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" --extended-lambda -std=c++17 -Xcompiler=-fPIC -c
                "${RAYD_REFLECTION_DEDUP_CU}"
                ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                ${RAYD_CUDA_DEFINITIONS_LIST}
                -o "${RAYD_REFLECTION_DEDUP_OBJECT}"
        DEPENDS
            "${RAYD_REFLECTION_DEDUP_CU}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_dedup.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYD_REFLECTION_DEDUP_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYD_CORE_SOURCE_FILES "${RAYD_REFLECTION_DEDUP_OBJECT}")

set(RAYD_REFLECTION_TRACE_CU "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_trace.cu")
set(RAYD_REFLECTION_TRACE_PTX "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace.ptx")
set(RAYD_REFLECTION_TRACE_PTX_H "${CMAKE_CURRENT_BINARY_DIR}/reflection_trace_ptx.h")

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
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_trace_params.h"
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
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_SOURCE_DIR}/multipath/reflection_trace_params.h"
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

set(RAYD_NANOBIND_ARGS NB_DOMAIN drjit)
if(RAYD_NANOBIND_NOMINSIZE)
    list(APPEND RAYD_NANOBIND_ARGS NOMINSIZE)
endif()
if(RAYD_ENABLE_LTO)
    list(APPEND RAYD_NANOBIND_ARGS LTO)
endif()

add_library(rayd_core STATIC ${RAYD_CORE_SOURCE_FILES})
target_include_directories(rayd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
)
target_compile_definitions(rayd_core PRIVATE
    RAYD_OPTIX_MODULE_OPT_LEVEL=${RAYD_OPTIX_MODULE_OPT_LEVEL}
    RAYD_OPTIX_EXCEPTION_FLAGS=${RAYD_OPTIX_EXCEPTION_FLAGS}
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

nanobind_add_module(rayd ${RAYD_NANOBIND_ARGS} ${RAYD_SOURCE_DIR}/rayd.cpp)
target_link_libraries(rayd PRIVATE rayd_core)
if(WIN32)
    target_compile_options(rayd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O2>
    )
endif()

install(TARGETS rayd DESTINATION rayd)
install(TARGETS rayd_core
    ARCHIVE DESTINATION rayd/lib
    LIBRARY DESTINATION rayd/lib
    RUNTIME DESTINATION rayd/bin
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rayd
    DESTINATION .
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.slang"
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rayd_slang.slang
    DESTINATION .
)

set_target_properties(rayd PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_property(TARGET rayd PROPERTY CXX_STANDARD 17)
set_target_properties(rayd PROPERTIES PREFIX "")

