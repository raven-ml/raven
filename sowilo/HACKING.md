# Sowilo Developer Guide

## Architecture

Sowilo provides computer vision operations built on Nx tensors. It implements image processing and CV algorithms using pure Nx operations where possible.

### Core Components

- **[lib/sowilo.ml](lib/sowilo.ml)**: All CV operations in a single module
- **[lib/sowilo.mli](lib/sowilo.mli)**: Public API interface

### Key Design Principles

1. **Nx-native**: Operations implemented via Nx tensor primitives
2. **NumPy compatibility**: Follow OpenCV/scikit-image conventions where reasonable
3. **Type safety**: Use dtype system to enforce image format requirements
4. **Zero-copy**: Leverage Nx views for crops and flips

## Image Representation

### Image Formats

Images are Nx tensors with shape:
- **Grayscale**: `[H; W]` or `[H; W; 1]`
- **RGB**: `[H; W; 3]`
- **RGBA**: `[H; W; 4]`

Channels: `[...; C]` where C = 1 (gray), 3 (RGB), 4 (RGBA)

### Data Types

Common dtypes:
- `uint8`: Standard 8-bit images (0-255)
- `float32`: Normalized images (0.0-1.0) for computation
- `float64`: High-precision computation

**Conversions:**
```ocaml
(* uint8 → float32: divide by 255 *)
let to_float img = Nx.div (Nx.cast Float32 img) (Nx.scalar Float32 255.)

(* float32 → uint8: multiply by 255, clip, cast *)
let to_uint8 img =
  let scaled = Nx.mul img (Nx.scalar Float32 255.) in
  let clipped = Nx.clip scaled ~min:0. ~max:255. in
  Nx.cast Uint8 clipped
```

## Development Workflow

### Building and Testing

```bash
# Build sowilo
dune build sowilo/

# Run tests
dune build sowilo/test/test_sowilo.exe && _build/default/sowilo/test/test_sowilo.exe

# Run examples
dune exec sowilo/example/edge_detection.exe
```

### Testing with Images

Tests and examples use Nx I/O:

```ocaml
open Nx
open Nx_io
open Sowilo

let img = load_image "input.jpg" in  (* Returns uint8 tensor *)
let gray = to_grayscale img in
let edges = canny ~low:50 ~high:150 gray in
save_image edges "edges.png"
```

**Visual testing:**
1. Load test image
2. Apply operation
3. Save output
4. Visually verify result

## Core Operations

### Basic Transforms

**Geometric:**
- `flip_vertical`, `flip_horizontal`: Zero-copy via Nx slicing
- `crop`: Extract region via Nx slicing
- `resize`: Nearest-neighbor or bilinear interpolation

**Color space:**
- `to_grayscale`: Weighted sum of RGB channels (0.299R + 0.587G + 0.114B)
- `rgb_to_bgr`, `bgr_to_rgb`: Channel reordering

### Filtering

**Convolution-based:**
- `gaussian_blur`: Create Gaussian kernel, convolve
- `median_blur`: Sliding window median (slow, consider optimization)
- `sobel`: Directional derivative kernels

**Implementation pattern:**

```ocaml
let gaussian_blur ~ksize ~sigma img =
  (* 1. Create kernel *)
  let kernel = make_gaussian_kernel ksize sigma in

  (* 2. Convolve *)
  convolve2d img kernel
```

### Edge Detection

**Canny algorithm:**
1. Gaussian blur (noise reduction)
2. Sobel gradients (magnitude and direction)
3. Non-maximum suppression
4. Double threshold
5. Edge tracking by hysteresis

**Sobel:**
- Compute gradient in X/Y directions
- Combine into magnitude: `sqrt(dx² + dy²)`

### Morphological Operations

- `erode`: Minimum filter with structuring element
- `dilate`: Maximum filter with structuring element

Pattern: Sliding window over image, apply operation with kernel.

## Adding Operations

### Pure Nx Operations

Prefer implementing via Nx primitives:

```ocaml
let brighten ~amount img =
  (* Add scalar to all pixels *)
  let amt = Nx.full (dtype img) (shape img) amount in
  Nx.add img amt |> Nx.clip ~min:0. ~max:255.
```

**Benefits:**
- Backend agnostic
- JIT-compilable
- Composable with Rune

### Custom Algorithms

When Nx primitives insufficient:

```ocaml
let median_blur ~ksize img =
  (* Extract patches → compute median → assemble result *)
  let h, w = ... in
  let out = Nx.zeros_like img in

  (* Sliding window over image *)
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      (* Extract window *)
      let window = extract_window img i j ksize in
      (* Compute median *)
      let med = compute_median window in
      (* Set output *)
      Nx.set out [|i; j|] med
    done
  done;
  out
```

**Optimize:**
- Use Nx slicing for window extraction
- Vectorize where possible
- Consider Rune JIT for tight loops

### Convolution

Core primitive for many operations:

```ocaml
let convolve2d img kernel =
  (* Implement 2D convolution *)
  (* For each output pixel: *)
  (*   sum(img[i+di, j+dj] * kernel[di, dj]) *)
  ...
```

**Padding:**
- `valid`: No padding, output smaller
- `same`: Pad to keep output size
- `full`: Pad to expand output

## Common Pitfalls

### Image Format Assumptions

Operations may require specific formats:

```ocaml
(* Wrong: pass RGB to grayscale operation *)
let edges = canny rgb_img ...  (* Error: expects 2D *)

(* Correct: convert first *)
let gray = to_grayscale rgb_img in
let edges = canny gray ...
```

### Dtype Conversions

uint8 operations can overflow:

```ocaml
(* Wrong: uint8 arithmetic overflows *)
let bright = Nx.add img (Nx.scalar Uint8 100)  (* Overflow! *)

(* Correct: convert to float, operate, convert back *)
let img_f = to_float img in
let bright_f = Nx.add img_f (Nx.scalar Float32 0.4) in
let bright = to_uint8 bright_f
```

### Channel Ordering

OpenCV uses BGR; most libraries use RGB:

```ocaml
(* Image loaded as RGB but algorithm expects BGR *)
let bgr_img = rgb_to_bgr img
```

### Out-of-Bounds Access

Filtering near edges requires padding or bounds checks:

```ocaml
(* Apply kernel without bounds check → crash *)

(* Correct: pad image or handle boundaries *)
let padded = Nx.pad img ~pad_width:[[k/2; k/2]; [k/2; k/2]] in
apply_kernel padded kernel
```

## Performance

- **Vectorize**: Use Nx operations over pixel loops
- **Avoid copies**: Use views (crop, flip) when possible
- **JIT compile**: Rune JIT for tight loops in CV pipelines
- **Batch operations**: Process multiple images in one tensor

## Testing

Visual tests via examples:

```ocaml
(* example/test_filter.ml *)
let () =
  let img = Nx_io.load_image "test.jpg" in
  let blur = gaussian_blur ~ksize:5 ~sigma:1.0 img in
  Nx_io.save_image blur "blur.png"
```

**Automated tests:**
- Test on synthetic images with known output
- Check edge cases (empty, single pixel, etc.)
- Verify dtype conversions

## Code Style

- **Naming**: `snake_case` for functions
- **Parameters**: Use labeled arguments for clarity (`~ksize`, `~sigma`)
- **Errors**: `"function_name: error description"`
- **Documentation**: Document expected image format and dtypes

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Nx tensor operations
- OpenCV documentation for algorithm reference
