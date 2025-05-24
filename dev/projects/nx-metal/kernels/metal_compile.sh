#!/bin/bash
# metal_compile.sh

# Make script exit on first error
set -e

# Function to compile metal file
compile_metal() {
    local shader=$1
    local target_dir=$2
    local std_version=$3
    local stem=$(basename "$shader" .metal)
    local target="${target_dir}/${stem}.air"
    
    xcrun -sdk macosx metal -c "$shader" \
          -std="$std_version" \
          -o "$target"
}

# Find all metal files
for shader in *.metal; do
    # Compile for Metal 3.0
    compile_metal "$shader" "air_basic" "metal3.0"
    
    # Compile for Metal 3.1
    compile_metal "$shader" "air_bfloat" "metal3.1"
done
