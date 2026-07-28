# OptiX SDK discovery and committed-PTX embedding helpers.

include_guard(GLOBAL)

# Expected in the caller's scope: CUDA_NVCC_EXECUTABLE, RAYD_MSVC_SETUP (Windows),
# RAYD_CUDA_INCLUDE_FLAGS(_LIST), RAYD_CUDA_DEFINITIONS(_LIST), RAYD_ROOT, and
# RAYD_GENERATED_PTX_DIR.

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

# Make one OptiX PTX blob available as a C++ header and append that header to the
# list named by OUT_SOURCES.
#
#   NAME        blob stem (<NAME>.ptx, build_<NAME>_ptx.bat, C array <NAME>_ptx)
#   SOURCE      absolute path of the .cu file
#   HEADER      committed/generated header filename under generated/drjit/ptx
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
        set(_committed "${RAYD_GENERATED_PTX_DIR}/${ARG_HEADER}")
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
    set(_header "${CMAKE_CURRENT_BINARY_DIR}/generated/drjit/ptx/${ARG_HEADER}")
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
            -P "${RAYD_ROOT}/cmake/embed_ptx.cmake"
        DEPENDS
            "${_ptx}"
            "${RAYD_ROOT}/cmake/embed_ptx.cmake"
        VERBATIM
    )
    set_source_files_properties("${_header}" PROPERTIES GENERATED TRUE)
    set(${ARG_OUT_SOURCES} ${${ARG_OUT_SOURCES}} "${_header}" PARENT_SCOPE)
endfunction()
