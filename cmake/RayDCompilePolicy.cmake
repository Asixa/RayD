# CUDA object compilation policy helpers shared by RayD backend frontends.

include_guard(GLOBAL)

# Expected in the caller's scope: CUDA_NVCC_EXECUTABLE, RAYD_MSVC_SETUP (Windows),
# RAYD_CUDA_GENCODE_FLAGS(_LIST), RAYD_CUDA_INCLUDE_FLAGS(_LIST), and
# RAYD_CUDA_DEFINITIONS(_LIST).

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
