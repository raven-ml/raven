# sowilo ᛋ Documentation

Sowilo is our OpenCV. It brings computer vision to the Raven ecosystem.

## What sowilo Does

Sowilo provides image processing operations that work with nx arrays and differentiate through Rune. Load images, apply filters, detect edges, transform geometries, all with automatic differentiation when you need it.

The name comes from the rune ᛋ meaning "sun." Computer vision brings clarity to visual data, like sunlight illuminating the world.

## Current Status

Sowilo implements core image processing operations:
- Image I/O (via stb_image)
- Basic filters (Gaussian blur, median, bilateral)
- Edge detection (Sobel, Canny)
- Morphological operations
- Color space conversions

What's missing: advanced CV algorithms (SIFT, object detection), video processing, and GPU acceleration. These come after v1.

## Design

Sowilo operations work on nx arrays, not custom image types. An image is just a 3D array with shape [height, width, channels]. This means you can mix image processing with any nx operation.

When Rune's JIT lands, sowilo operations will compile to efficient GPU kernels automatically.

## Learn More

- [Getting Started](/docs/sowilo/getting-started/) - Image processing basics
- [Examples](https://github.com/raven-ml/raven/tree/main/sowilo/example) - Image processing examples
