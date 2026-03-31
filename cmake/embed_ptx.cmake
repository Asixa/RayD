file(READ "${PTX_FILE}" PTX_CONTENT)
string(LENGTH "${PTX_CONTENT}" PTX_SIZE)
file(WRITE "${OUTPUT_FILE}"
    "// Auto-generated. Do not edit.\n"
    "#pragma once\n"
    "#include <cstddef>\n"
    "static const char ${VAR_NAME}[] = R\"PTX(\n"
    "${PTX_CONTENT}"
    ")PTX\";\n"
    "static const size_t ${VAR_NAME}_size = ${PTX_SIZE};\n"
)
