cmake_minimum_required(VERSION 3.26)
project(rayd_workspace LANGUAGES NONE)

option(RAYD_BUILD_DRJIT "Configure the Dr.Jit backend." OFF)
option(RAYD_BUILD_TORCH "Configure the Torch backend." OFF)

if(RAYD_BUILD_DRJIT)
    add_subdirectory(drjit)
endif()

if(RAYD_BUILD_TORCH)
    add_subdirectory(torch)
endif()

if(NOT RAYD_BUILD_DRJIT AND NOT RAYD_BUILD_TORCH)
    message(STATUS "Enable RAYD_BUILD_DRJIT and/or RAYD_BUILD_TORCH, or build a backend directly.")
endif()

