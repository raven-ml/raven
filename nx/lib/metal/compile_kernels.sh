#!/bin/bash
# Compile Metal kernels and embed them into a C file

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KERNELS_DIR="$SCRIPT_DIR/kernels"
OUTPUT_C="$SCRIPT_DIR/kernels_data_generated.c"

# Check if kernels directory exists
if [ ! -d "$KERNELS_DIR" ]; then
    echo "Error: Kernels directory not found at $KERNELS_DIR"
    exit 1
fi

# Concatenate all Metal source files
cat $KERNELS_DIR/*.metal > "$SCRIPT_DIR/all_kernels.metal"

# Convert to C string
echo '#include <caml/mlvalues.h>' > "$OUTPUT_C"
echo '#include <caml/alloc.h>' >> "$OUTPUT_C"
echo '#include <caml/memory.h>' >> "$OUTPUT_C"
echo '#include <string.h>' >> "$OUTPUT_C"
echo '' >> "$OUTPUT_C"
echo 'static const char metal_kernels_source[] = ' >> "$OUTPUT_C"

# Convert Metal source to C string literal
sed 's/\\/\\\\/g; s/"/\\"/g; s/^/"/; s/$/\\n"/' "$SCRIPT_DIR/all_kernels.metal" >> "$OUTPUT_C"

echo ';' >> "$OUTPUT_C"
echo '' >> "$OUTPUT_C"
echo 'CAMLprim value metal_kernels_data(value unit) {' >> "$OUTPUT_C"
echo '    CAMLparam1(unit);' >> "$OUTPUT_C"
echo '    CAMLlocal1(result);' >> "$OUTPUT_C"
echo '    result = caml_alloc_string(sizeof(metal_kernels_source) - 1);' >> "$OUTPUT_C"
echo '    memcpy(Bytes_val(result), metal_kernels_source, sizeof(metal_kernels_source) - 1);' >> "$OUTPUT_C"
echo '    CAMLreturn(result);' >> "$OUTPUT_C"
echo '}' >> "$OUTPUT_C"

# Clean up
rm "$SCRIPT_DIR/all_kernels.metal"

echo "Generated $OUTPUT_C"