#!/bin/bash
# Simple build script for Raven documentation site using soupault

set -e

echo "Building Raven documentation site..."

# Check if soupault is installed
if ! command -v soupault &> /dev/null; then
    echo "Error: soupault not found. Please install it first:"
    echo "  opam install soupault"
    exit 1
fi

# Clean build directory
rm -rf build
mkdir -p build

# Run soupault to build the site
cd /Users/tmattio/Workspace/raven/www
soupault

echo "Site built successfully in build/ directory"
echo "To serve locally: cd build && python3 -m http.server 8000"