# Source-build CUDA target selection and the installed build-target record.

include_guard(GLOBAL)

# RAYD_CUDA_ARCHS selects the GPUs a source build targets in both backends. It
# is either `native` (every distinct GPU nvidia-smi reports) or a list such as
# `61,89` or `6.1;8.9+PTX`, separated by commas, semicolons, or spaces. Every
# entry receives native SASS; a `+PTX` suffix also embeds forward-compatible
# PTX for that entry. The environment variable wins over the cache entry.
#
# sm_70 is the oldest architecture RayD's device code compiles for.
set(RAYD_MIN_CUDA_ARCH 70)
set(RAYD_CUDA_ARCHS "" CACHE STRING
    "Source-build CUDA targets: 'native' or a list such as 61,89 or 6.1;8.9+PTX.")
if(DEFINED ENV{RAYD_CUDA_ARCHS} AND NOT "$ENV{RAYD_CUDA_ARCHS}" STREQUAL "")
    set(RAYD_CUDA_ARCHS "$ENV{RAYD_CUDA_ARCHS}" CACHE STRING
        "Source-build CUDA targets: 'native' or a list such as 61,89 or 6.1;8.9+PTX." FORCE)
endif()

# Distinct compute capabilities of the local GPUs, as integers such as 89.
function(rayd_native_cuda_archs OUT_VAR)
    find_program(RAYD_NVIDIA_SMI_EXECUTABLE NAMES nvidia-smi)
    if(NOT RAYD_NVIDIA_SMI_EXECUTABLE)
        message(FATAL_ERROR "RAYD_CUDA_ARCHS=native needs nvidia-smi to detect the local GPU. "
            "Pass an explicit list instead, for example RAYD_CUDA_ARCHS=61.")
    endif()
    execute_process(
        COMMAND "${RAYD_NVIDIA_SMI_EXECUTABLE}" --query-gpu=compute_cap --format=csv,noheader
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCHALL "[0-9]+\\.[0-9]+" _capabilities "${_output}")
    if(NOT _result EQUAL 0 OR NOT _capabilities)
        message(FATAL_ERROR "RAYD_CUDA_ARCHS=native could not read a compute capability from nvidia-smi. "
            "Pass an explicit list instead, for example RAYD_CUDA_ARCHS=61.")
    endif()
    set(_archs "")
    foreach(_capability IN LISTS _capabilities)
        string(REPLACE "." "" _arch "${_capability}")
        list(APPEND _archs "${_arch}")
    endforeach()
    list(REMOVE_DUPLICATES _archs)
    set(${OUT_VAR} "${_archs}" PARENT_SCOPE)
endfunction()

# Parse RAYD_CUDA_ARCHS. OUT_SET is TRUE when it was given; OUT_SASS and
# OUT_PTX receive sorted integer lists.
function(rayd_parse_cuda_archs OUT_SET OUT_SASS OUT_PTX)
    string(STRIP "${RAYD_CUDA_ARCHS}" _spec)
    if(_spec STREQUAL "")
        set(${OUT_SET} FALSE PARENT_SCOPE)
        set(${OUT_SASS} "" PARENT_SCOPE)
        set(${OUT_PTX} "" PARENT_SCOPE)
        return()
    endif()
    string(REGEX REPLACE "[,; \t]+" ";" _entries "${_spec}")
    set(_sass "")
    set(_ptx "")
    foreach(_entry IN LISTS _entries)
        if(_entry STREQUAL "")
            continue()
        endif()
        string(TOLOWER "${_entry}" _entry)
        if(_entry STREQUAL "native")
            rayd_native_cuda_archs(_native)
            list(APPEND _sass ${_native})
            continue()
        endif()
        if(NOT _entry MATCHES "^([0-9]+)(\\.([0-9]))?(\\+ptx)?$")
            message(FATAL_ERROR "Invalid RAYD_CUDA_ARCHS entry '${_entry}'. "
                "Use 'native' or compute capabilities such as 61, 6.1, or 8.9+PTX.")
        endif()
        if(CMAKE_MATCH_2)
            set(_arch "${CMAKE_MATCH_1}${CMAKE_MATCH_3}")
        else()
            set(_arch "${CMAKE_MATCH_1}")
        endif()
        if(_arch LESS RAYD_MIN_CUDA_ARCH)
            message(FATAL_ERROR "RAYD_CUDA_ARCHS entry sm_${_arch} is below sm_${RAYD_MIN_CUDA_ARCH}. "
                "RayD device code uses Volta-only warp matching (__match_any_sync) and "
                "__grid_constant__ kernel parameters, so it cannot be built for older GPUs.")
        endif()
        list(APPEND _sass "${_arch}")
        if(CMAKE_MATCH_4)
            list(APPEND _ptx "${_arch}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _sass)
    list(REMOVE_DUPLICATES _ptx)
    list(SORT _sass COMPARE NATURAL)
    list(SORT _ptx COMPARE NATURAL)
    set(${OUT_SET} TRUE PARENT_SCOPE)
    set(${OUT_SASS} "${_sass}" PARENT_SCOPE)
    set(${OUT_PTX} "${_ptx}" PARENT_SCOPE)
endfunction()

# Translate a CMAKE_CUDA_ARCHITECTURES value into SASS and PTX lists. OUT_KNOWN
# is FALSE for values that name no concrete list, such as `all`.
function(rayd_cuda_targets_from_architectures VALUE OUT_KNOWN OUT_SASS OUT_PTX)
    set(_value "${VALUE}")
    if(_value STREQUAL "native")
        if(NOT CMAKE_CUDA_ARCHITECTURES_NATIVE)
            set(${OUT_KNOWN} FALSE PARENT_SCOPE)
            return()
        endif()
        set(_value "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    endif()
    set(_sass "")
    set(_ptx "")
    foreach(_entry IN LISTS _value)
        if(NOT _entry MATCHES "^([0-9]+)(-real|-virtual)?$")
            set(${OUT_KNOWN} FALSE PARENT_SCOPE)
            return()
        endif()
        if(NOT CMAKE_MATCH_2 STREQUAL "-virtual")
            list(APPEND _sass "${CMAKE_MATCH_1}")
        endif()
        if(NOT CMAKE_MATCH_2 STREQUAL "-real")
            list(APPEND _ptx "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    set(${OUT_KNOWN} TRUE PARENT_SCOPE)
    set(${OUT_SASS} "${_sass}" PARENT_SCOPE)
    set(${OUT_PTX} "${_ptx}" PARENT_SCOPE)
endfunction()

# Translate explicit nvcc --generate-code/-gencode flags into SASS and PTX lists.
function(rayd_cuda_targets_from_gencode FLAGS OUT_SASS OUT_PTX)
    string(REGEX MATCHALL "code=(\\[[^]]*\\]|[a-z_0-9,]+)" _codes "${FLAGS}")
    set(_sass "")
    set(_ptx "")
    foreach(_code IN LISTS _codes)
        string(REGEX MATCHALL "(sm|compute)_[0-9]+" _targets "${_code}")
        foreach(_target IN LISTS _targets)
            string(REGEX MATCH "[0-9]+" _arch "${_target}")
            if(_target MATCHES "^sm_")
                list(APPEND _sass "${_arch}")
            else()
                list(APPEND _ptx "${_arch}")
            endif()
        endforeach()
    endforeach()
    set(${OUT_SASS} "${_sass}" PARENT_SCOPE)
    set(${OUT_PTX} "${_ptx}" PARENT_SCOPE)
endfunction()

# Write the `_build_config.json` record read by rayd/_impl/device_support.py.
# KNOWN is FALSE when the CUDA kernels used the compiler default architecture.
function(rayd_write_build_config PATH BACKEND KNOWN SASS PTX OPTIX)
    list(REMOVE_ITEM SASS "")
    list(REMOVE_ITEM PTX "")
    if(KNOWN)
        list(REMOVE_DUPLICATES SASS)
        list(SORT SASS COMPARE NATURAL)
        string(REPLACE ";" ", " _sass "[${SASS}]")
    else()
        set(_sass "null")
    endif()
    list(REMOVE_DUPLICATES PTX)
    list(SORT PTX COMPARE NATURAL)
    string(REPLACE ";" ", " _ptx "[${PTX}]")
    file(CONFIGURE OUTPUT "${PATH}" CONTENT
"{\"backend\": \"${BACKEND}\", \"cuda_sass\": ${_sass}, \"cuda_ptx\": ${_ptx}, \"optix_ptx\": ${OPTIX}}
" @ONLY)
endfunction()
