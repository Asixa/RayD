# Shared helpers for the Dr.Jit backend's explicit nvcc invocations.
#
# The backend compiles its CUDA translation units and its OptiX PTX blobs with
# hand-written nvcc command lines instead of CMake's CUDA language so the emitted
# flag set stays exactly under this repository's control. These helpers own the
# two command shapes; call sites supply only the per-unit inputs.
#
# On Windows nvcc runs through a generated .bat wrapper so the MSVC environment
# (RAYD_MSVC_SETUP) can be initialized first; on POSIX the command runs directly.
# The two variants must stay in sync.
#
# Expected in the caller's scope: CUDA_NVCC_EXECUTABLE, RAYD_MSVC_SETUP (Windows),
# RAYD_CUDA_GENCODE_FLAGS(_LIST), RAYD_CUDA_INCLUDE_FLAGS(_LIST),
# RAYD_CUDA_DEFINITIONS(_LIST), and RAYD_INCLUDE_DIR.

# Locate the OptiX SDK headers needed to regenerate a PTX blob.
# CONTEXT names the blob and OPTION_NAME the regenerate option, both only for the
# failure message. OPTIX_INCLUDE_DIR stays a user-overridable cache entry.
function(rayd_find_optix_include_dir OUT_VAR CONTEXT OPTION_NAME)
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
            "Could not locate OptiX SDK headers needed to regenerate ${CONTEXT} PTX. "
            "Set OPTIX_INCLUDE_DIR or OPTIX_PATH, or disable ${OPTION_NAME}.")
    endif()
    set(${OUT_VAR} "${OPTIX_INCLUDE_DIR}" PARENT_SCOPE)
endfunction()

# Compile one CUDA translation unit to an object file and append the object to
# the list named by OUT_SOURCES.
#
#   NAME                     object/script stem (<NAME>.obj, build_<NAME>.bat)
#   SOURCE                   absolute path of the .cu file
#   EXTRA_FLAGS              extra nvcc flags placed after --extended-lambda
#   DEPENDS                  header dependencies; SOURCE is always added first
#   OUT_SOURCES              name of the source list to append the object to
#   POSIX_NO_EXTENDED_LAMBDA drop --extended-lambda on POSIX only
#
# POSIX_NO_EXTENDED_LAMBDA preserves a pre-existing asymmetry: the six edge/BVH
# units pass --extended-lambda on Windows but not on POSIX. Unifying it would
# change the released Linux device code, so it stays until it can be proven
# equivalent on a Linux builder.
function(rayd_cuda_object)
    cmake_parse_arguments(PARSE_ARGV 0 ARG
        "POSIX_NO_EXTENDED_LAMBDA"
        "NAME;SOURCE;OUT_SOURCES"
        "EXTRA_FLAGS;DEPENDS")

    if(WIN32)
        set(_object "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}.obj")
        set(_script "${CMAKE_CURRENT_BINARY_DIR}/build_${ARG_NAME}.bat")
        string(JOIN " " _flags --extended-lambda ${ARG_EXTRA_FLAGS})
        file(GENERATE OUTPUT "${_script}" CONTENT
"@echo off\r\n\
${RAYD_MSVC_SETUP}\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" ${_flags} -std=c++17 -c \"${ARG_SOURCE}\" ${RAYD_CUDA_GENCODE_FLAGS} ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -Xcompiler \"/MD /O2 /EHsc /wd4819\" -o \"${_object}\"\r\n\
")
        add_custom_command(
            OUTPUT "${_object}"
            COMMAND "${_script}"
            DEPENDS "${_script}" "${ARG_SOURCE}" ${ARG_DEPENDS}
            VERBATIM
        )
    else()
        set(_object "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}.o")
        if(ARG_POSIX_NO_EXTENDED_LAMBDA)
            set(_flags ${ARG_EXTRA_FLAGS})
        else()
            set(_flags --extended-lambda ${ARG_EXTRA_FLAGS})
        endif()
        add_custom_command(
            OUTPUT "${_object}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ccbin "${CMAKE_CXX_COMPILER}" ${_flags} -std=c++17 -Xcompiler=-fPIC -c
                    "${ARG_SOURCE}"
                    ${RAYD_CUDA_GENCODE_FLAGS_LIST}
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${_object}"
            DEPENDS "${ARG_SOURCE}" ${ARG_DEPENDS}
            VERBATIM
        )
    endif()

    set_source_files_properties("${_object}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(${ARG_OUT_SOURCES} ${${ARG_OUT_SOURCES}} "${_object}" PARENT_SCOPE)
endfunction()

# Make one OptiX PTX blob available as a C++ header and append that header to the
# list named by OUT_SOURCES.
#
#   NAME        blob stem (<NAME>.ptx, build_<NAME>_ptx.bat, C array <NAME>_ptx)
#   SOURCE      absolute path of the .cu file
#   HEADER      header path relative to include/rayd (and to the binary rayd dir)
#   OPTION      name of the RAYD_REGENERATE_*_PTX option guarding regeneration
#   DEPENDS     header dependencies; SOURCE is always added first
#   OUT_SOURCES name of the source list to append the header to
#
# With the option off (the default, and what wheels build) the committed header is
# used verbatim and no OptiX SDK is required.
function(rayd_embed_ptx)
    cmake_parse_arguments(PARSE_ARGV 0 ARG
        ""
        "NAME;SOURCE;HEADER;OPTION;OUT_SOURCES"
        "DEPENDS")

    if(NOT ${ARG_OPTION})
        set(_committed "${CMAKE_CURRENT_SOURCE_DIR}/${RAYD_INCLUDE_DIR}/${ARG_HEADER}")
        if(NOT EXISTS "${_committed}")
            message(FATAL_ERROR
                "Missing committed ${ARG_NAME} PTX header at "
                "${_committed}. "
                "Restore it or enable ${ARG_OPTION} with OptiX SDK headers available.")
        endif()
        set(${ARG_OUT_SOURCES} ${${ARG_OUT_SOURCES}} "${_committed}" PARENT_SCOPE)
        return()
    endif()

    rayd_find_optix_include_dir(_optix_include_dir "${ARG_NAME}" "${ARG_OPTION}")

    set(_ptx "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}.ptx")
    set(_header "${CMAKE_CURRENT_BINARY_DIR}/rayd/${ARG_HEADER}")
    if(WIN32)
        set(_script "${CMAKE_CURRENT_BINARY_DIR}/build_${ARG_NAME}_ptx.bat")
        file(GENERATE OUTPUT "${_script}" CONTENT
"@echo off\r\n\
${RAYD_MSVC_SETUP}\
if errorlevel 1 exit /b %errorlevel%\r\n\
\"${CUDA_NVCC_EXECUTABLE}\" -ptx --use_fast_math -std=c++17 -arch=compute_70 \"${ARG_SOURCE}\" -I\"${_optix_include_dir}\" ${RAYD_CUDA_INCLUDE_FLAGS} ${RAYD_CUDA_DEFINITIONS} -o \"${_ptx}\"\r\n\
")
        add_custom_command(
            OUTPUT "${_ptx}"
            COMMAND "${_script}"
            DEPENDS "${_script}" "${ARG_SOURCE}" ${ARG_DEPENDS}
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${_ptx}"
            COMMAND "${CUDA_NVCC_EXECUTABLE}" -ptx --use_fast_math -std=c++17 -arch=compute_70
                    "${ARG_SOURCE}"
                    -I"${_optix_include_dir}"
                    ${RAYD_CUDA_INCLUDE_FLAGS_LIST}
                    ${RAYD_CUDA_DEFINITIONS_LIST}
                    -o "${_ptx}"
            DEPENDS "${ARG_SOURCE}" ${ARG_DEPENDS}
            VERBATIM
        )
    endif()

    add_custom_command(
        OUTPUT "${_header}"
        COMMAND ${CMAKE_COMMAND}
            -DPTX_FILE=${_ptx}
            -DOUTPUT_FILE=${_header}
            -DVAR_NAME=${ARG_NAME}_ptx
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        DEPENDS
            "${_ptx}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${_header}" PROPERTIES GENERATED TRUE)
    set(${ARG_OUT_SOURCES} ${${ARG_OUT_SOURCES}} "${_header}" PARENT_SCOPE)
endfunction()
