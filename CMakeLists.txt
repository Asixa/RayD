cmake_minimum_required(VERSION 3.15...3.22)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RAYDI_OPTIX_PRODUCTION_CONFIG "Use OptiX production module settings (LEVEL_3 + no exceptions)." ON)
option(RAYDI_NANOBIND_NOMINSIZE "Disable nanobind's size-oriented optimization flags in optimized configs." ON)
option(RAYDI_ENABLE_LTO "Enable nanobind/C++ link-time optimization." OFF)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import drjit, sys; sys.stdout.write(drjit.get_cmake_dir())"
    RESULT_VARIABLE DRJIT_CMAKE_DIR_RESULT
    OUTPUT_VARIABLE DRJIT_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT DRJIT_CMAKE_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate the installed Dr.Jit CMake package. Ensure drjit>=1.3.0 is installed in the build environment.")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind, sys; sys.stdout.write(nanobind.cmake_dir())"
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

if(WIN32)
    add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-reorder -Wno-sign-compare -fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -DNDEBUG")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(RAYDI_OPTIX_PRODUCTION_CONFIG)
    set(RAYDI_OPTIX_MODULE_OPT_LEVEL 0x2343)
    set(RAYDI_OPTIX_EXCEPTION_FLAGS 0)
else()
    set(RAYDI_OPTIX_MODULE_OPT_LEVEL 0x2340)
    set(RAYDI_OPTIX_EXCEPTION_FLAGS 11)
endif()

find_package(CUDA 11.0 REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include_directories(
    include/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set(RAYDI_INCLUDE_DIR include/raydi)
set(RAYDI_SOURCE_DIR src)
set(RAYDI_SOURCE_FILES
    ${RAYDI_INCLUDE_DIR}/raydi.h
    ${RAYDI_INCLUDE_DIR}/constants.h
    ${RAYDI_INCLUDE_DIR}/fwd.h
    ${RAYDI_INCLUDE_DIR}/types.h
    ${RAYDI_INCLUDE_DIR}/utils.h

    ${RAYDI_INCLUDE_DIR}/ray.h
    ${RAYDI_INCLUDE_DIR}/intersection.h
    ${RAYDI_INCLUDE_DIR}/transform.h

    ${RAYDI_INCLUDE_DIR}/edge.h

    ${RAYDI_INCLUDE_DIR}/mesh.h
    ${RAYDI_SOURCE_DIR}/mesh.cpp

    ${RAYDI_INCLUDE_DIR}/optix.h

    ${RAYDI_INCLUDE_DIR}/camera.h
    ${RAYDI_SOURCE_DIR}/camera.cpp

    ${RAYDI_INCLUDE_DIR}/scene/scene_optix.h
    ${RAYDI_SOURCE_DIR}/scene/scene_optix.cpp
    ${RAYDI_INCLUDE_DIR}/scene/scene_edge.h
    ${RAYDI_SOURCE_DIR}/scene/scene_edge.cpp
    ${RAYDI_SOURCE_DIR}/scene/edge_bvh.h

    ${RAYDI_INCLUDE_DIR}/scene/scene.h
    ${RAYDI_SOURCE_DIR}/scene/scene.cpp

    ${RAYDI_SOURCE_DIR}/optix.cpp
    ${RAYDI_SOURCE_DIR}/raydi.cpp
)
if(WIN32)
    set(RAYDI_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.obj")
    set(RAYDI_VSDEVCMD "${CMAKE_GENERATOR_INSTANCE}/Common7/Tools/VsDevCmd.bat")
    set(RAYDI_CUDA_BUILD_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_edge_bvh.bat")
    file(GENERATE OUTPUT "${RAYDI_CUDA_BUILD_SCRIPT}" CONTENT
"@echo off\r\n\
call \"${RAYDI_VSDEVCMD}\" -arch=x64\r\n\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" --extended-lambda -std=c++17 -c \"${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.cu\" -I\"${CMAKE_CURRENT_SOURCE_DIR}/include\" -I\"${CMAKE_CURRENT_SOURCE_DIR}\" -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${RAYDI_CUDA_OBJECT}\"\r\n\
")
    add_custom_command(
        OUTPUT "${RAYDI_CUDA_OBJECT}"
        COMMAND "${RAYDI_CUDA_BUILD_SCRIPT}"
        DEPENDS
            "${RAYDI_CUDA_BUILD_SCRIPT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.h"
        VERBATIM
    )
else()
    set(RAYDI_CUDA_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/edge_bvh.o")
    add_custom_command(
        OUTPUT "${RAYDI_CUDA_OBJECT}"
        COMMAND "${CUDA_NVCC_EXECUTABLE}" -std=c++17 -Xcompiler=-fPIC -c
                "${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.cu"
                -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
                -I"${CMAKE_CURRENT_SOURCE_DIR}"
                -o "${RAYDI_CUDA_OBJECT}"
        DEPENDS
            "${RAYDI_CUDA_BUILD_SCRIPT}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/${RAYDI_SOURCE_DIR}/scene/edge_bvh.h"
        VERBATIM
    )
endif()
set_source_files_properties("${RAYDI_CUDA_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND RAYDI_SOURCE_FILES "${RAYDI_CUDA_OBJECT}")

set(RAYDI_NANOBIND_ARGS NB_DOMAIN drjit)
if(RAYDI_NANOBIND_NOMINSIZE)
    list(APPEND RAYDI_NANOBIND_ARGS NOMINSIZE)
endif()
if(RAYDI_ENABLE_LTO)
    list(APPEND RAYDI_NANOBIND_ARGS LTO)
endif()

nanobind_add_module(raydi ${RAYDI_NANOBIND_ARGS} ${RAYDI_SOURCE_FILES})
target_compile_definitions(raydi PRIVATE
    RAYDI_OPTIX_MODULE_OPT_LEVEL=${RAYDI_OPTIX_MODULE_OPT_LEVEL}
    RAYDI_OPTIX_EXCEPTION_FLAGS=${RAYDI_OPTIX_EXCEPTION_FLAGS}
)

if(WIN32)
    target_compile_options(raydi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/O2>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MinSizeRel>>:/O2>
    )
endif()

set(PYRAYDI_LIBRARIES drjit drjit-core drjit-extra nanothread)
if(WIN32)
    list(APPEND PYRAYDI_LIBRARIES version)
endif()
target_link_libraries(raydi PRIVATE ${PYRAYDI_LIBRARIES} ${CUDA_LIBRARIES})

install(TARGETS raydi DESTINATION raydi)

set_target_properties(raydi PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_property(TARGET raydi PROPERTY CXX_STANDARD 17)
set_target_properties(raydi PROPERTIES PREFIX "")

