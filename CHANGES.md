# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0~alpha0] - 2025-06-29

### Initial Alpha Release

We're excited to release the zeroth alpha of Raven, an OCaml machine learning ecosystem bringing modern scientific computing to OCaml.

### Added

#### Core Libraries

- **Nx** - N-dimensional array library with NumPy-like API
  - Multi-dimensional tensors with support for several data types.
  - Zero-copy operations: slicing, reshaping, broadcasting
  - Element-wise and linear algebra operations
  - Swappable backends: Native OCaml, CBLAS, Metal
  - I/O support for images (PNG, JPEG) and NumPy files (.npy, .npz)

- **Hugin** - Publication-quality plotting library
  - 2D plots: line, scatter, bar, histogram, step, error bars, fill-between
  - 3D plots: line3d, scatter3d
  - Image visualization: imshow, matshow
  - Contour plots with customizable levels
  - Text annotations and legends

- **Quill** - Interactive notebook environment
  - Markdown-based notebooks with live formatting
  - OCaml code execution with persistent session state
  - Integrated data visualization via Hugin
  - Web server mode for browser-based editing

#### ML/AI Components

- **Rune** - Automatic differentiation and JIT compilation framework
  - Reverse-mode automatic differentiation
  - Functional API for pure computations
  - Basic JIT infrastructure (in development)

- **Kaun** - Deep learning framework (experimental)
  - Flax-inspired functional API
  - Basic neural network components
  - Example implementations for XOR and MNIST

- **Sowilo** - Computer vision library
  - Image manipulation: flip, crop, color conversions
  - Filtering: gaussian_blur, median_blur
  - Morphological operations and edge detection

#### Supporting Libraries

- **Nx-datasets** - Common ML datasets (MNIST, Iris, California Housing)
- **Nx-text** - Text processing and tokenization utilities

### Known Issues

This is an alpha release with several limitations:
- Quill editor has UI bugs being addressed
- APIs may change significantly before stable release

### Contributors

Initial development by the Raven team. Special thanks to all early testers and contributors.

@axrwl
@gabyfle
@hesterjeng
@ghennequin
@blueavee

And to our early sponsors:

@daemonfire300
@gabyfle
@sabine

[1.0.0~alpha0]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha0
