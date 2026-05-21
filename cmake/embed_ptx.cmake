file(READ "${PTX_FILE}" PTX_CONTENT)
string(LENGTH "${PTX_CONTENT}" PTX_SIZE)
get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_FILE}"
    "// Auto-generated. Do not edit.\n"
    "#pragma once\n"
    "#include <cstddef>\n"
    "static const char ${VAR_NAME}[] =\n"
)

set(PTX_CHUNK_SIZE 16000)
set(PTX_OFFSET 0)
while(PTX_OFFSET LESS PTX_SIZE)
    math(EXPR PTX_REMAINING "${PTX_SIZE} - ${PTX_OFFSET}")
    if(PTX_REMAINING GREATER PTX_CHUNK_SIZE)
        set(PTX_CURRENT_CHUNK_SIZE ${PTX_CHUNK_SIZE})
    else()
        set(PTX_CURRENT_CHUNK_SIZE ${PTX_REMAINING})
    endif()
    string(SUBSTRING "${PTX_CONTENT}" ${PTX_OFFSET} ${PTX_CURRENT_CHUNK_SIZE} PTX_CHUNK)
    file(APPEND "${OUTPUT_FILE}" "R\"PTX_CHUNK(${PTX_CHUNK})PTX_CHUNK\"\n")
    math(EXPR PTX_OFFSET "${PTX_OFFSET} + ${PTX_CURRENT_CHUNK_SIZE}")
endwhile()

file(APPEND "${OUTPUT_FILE}"
    ";\n"
    "static const size_t ${VAR_NAME}_size = ${PTX_SIZE};\n"
)
