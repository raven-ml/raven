# Sowilo vs. OpenCV -- A Practical Comparison

This guide explains how Sowilo's image processing model relates to Python's [OpenCV](https://docs.opencv.org/), focusing on:

* How core concepts map (images, color spaces, filtering, morphology, edges)
* Where the APIs feel similar vs. deliberately different
* How to translate common OpenCV patterns into Sowilo

If you already use OpenCV, this should be enough to become productive in Sowilo quickly.

---

## 1. Big-Picture Differences

| Aspect           | OpenCV (Python)                                      | Sowilo (OCaml)                                                |
| ---------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| Language         | C++ core with Python bindings                        | Pure OCaml on Nx tensors                                      |
| Image type       | `numpy.ndarray`                                      | `Nx.t` (same type used everywhere in raven)                   |
| Channel order    | BGR by default                                       | RGB, channels-last `[H; W; C]`                                |
| Pixel range      | uint8 `[0, 255]` or float32/64                       | float32 `[0, 1]` (convert with `to_float` / `to_uint8`)       |
| Color conversion | `cv2.cvtColor` with 200+ codes                       | Named functions: `to_grayscale`, `rgb_to_hsv`, `hsv_to_rgb`   |
| Autodiff         | Not available                                        | All ops (except `median_blur`, `canny`) work with `Rune.grad` |
| Batching         | Manual loops or `np.stack`                           | Native batch dimension `[N; H; W; C]` + `Rune.vmap`           |
| Backend          | Optimized C++/CUDA                                   | Nx C backend (CPU)                                            |
| Mutability       | Arrays mutated in-place by convention                | Immutable tensors; operations return new `Nx.t`               |
| Scope            | Full vision library (video, GUI, ML, features, etc.) | Image processing primitives for ML pipelines                  |

**Sowilo semantics to know (read once):**
- Images are plain `Nx.t` tensors, not a separate type. Any Nx operation works on them.
- All operations expect float32 in `[0, 1]`. Use `to_float` to convert from uint8.
- Channel layout is always channels-last: `[H; W; C]` or `[N; H; W; C]` for batches.
- Every operation (except `median_blur` and `canny`) is differentiable through `Rune.grad`.

---

## 2. Image Representation

### 2.1 Loading and layout

**OpenCV**

```python
import cv2
import numpy as np

img = cv2.imread("photo.jpg")          # BGR, uint8, shape (H, W, 3)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_f = img_rgb.astype(np.float32) / 255.0
```

**Sowilo**

Sowilo does not provide I/O. Load with any image library that produces an `Nx.t`, then convert:

<!-- $MDX skip -->
```ocaml
(* Assuming img is a uint8 [H; W; C] tensor loaded from disk *)
let img = Sowilo.to_float img   (* float32 [0, 1], RGB, [H; W; C] *)
```

Key differences:

* OpenCV defaults to BGR ordering. Sowilo always uses RGB.
* OpenCV operates on uint8 or float64 interchangeably. Sowilo expects float32 in `[0, 1]` for all processing functions.
* There is no `cv2.imread` equivalent -- image I/O is outside Sowilo's scope.

### 2.2 Converting back to uint8

**OpenCV**

```python
out = (img_f * 255).clip(0, 255).astype(np.uint8)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let out = Sowilo.to_uint8 img   (* clips to [0, 1], scales to [0, 255], casts to uint8 *)
```

---

## 3. Type Conversion and Preprocessing

### 3.1 Normalization

**OpenCV / NumPy**

```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (img_f - mean) / std
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let normalized =
  Sowilo.normalize
    ~mean:[0.485; 0.456; 0.406]
    ~std:[0.229; 0.224; 0.225]
    img
```

Both apply per-channel `(img - mean) / std`. Sowilo raises `Invalid_argument` if the list lengths do not match the channel count.

### 3.2 Thresholding

**OpenCV**

```python
_, binary = cv2.threshold(gray, 0.5, 1.0, cv2.THRESH_BINARY)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let binary = Sowilo.threshold 0.5 gray
```

Sowilo's `threshold` returns `1.0` where `img > t`, `0.0` elsewhere. OpenCV's `cv2.threshold` has many modes (binary, truncate, adaptive, Otsu); Sowilo provides only the simple binary variant.

---

## 4. Color Space Conversion

**OpenCV**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb  = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let gray = Sowilo.to_grayscale img
let hsv  = Sowilo.rgb_to_hsv img
let rgb  = Sowilo.hsv_to_rgb hsv
```

Differences:

* OpenCV has 200+ conversion codes (`COLOR_BGR2Lab`, `COLOR_YUV2RGB_NV21`, etc.). Sowilo provides three conversions: grayscale, RGB-to-HSV, and HSV-to-RGB.
* OpenCV's HSV uses H in `[0, 180]`, S and V in `[0, 255]` for uint8. Sowilo normalizes all channels to `[0, 1]`.
* `to_grayscale` uses ITU-R BT.601 weights (`0.299 * R + 0.587 * G + 0.114 * B`), same as OpenCV's `COLOR_RGB2GRAY`.

---

## 5. Image Adjustments

OpenCV has no built-in brightness/contrast/saturation functions. The standard approach is manual arithmetic. Sowilo provides dedicated functions for these.

### 5.1 Brightness

**OpenCV**

```python
bright = np.clip(img_f * 1.5, 0, 1)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let bright = Sowilo.adjust_brightness 1.5 img
```

### 5.2 Contrast

**OpenCV**

```python
mean = img_f.mean(axis=(0, 1), keepdims=True)
contrasted = np.clip(mean + 1.5 * (img_f - mean), 0, 1)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let contrasted = Sowilo.adjust_contrast 1.5 img
```

A factor of `0` produces solid gray, `1` is the original image.

### 5.3 Saturation

**OpenCV**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let saturated = Sowilo.adjust_saturation 1.5 img
```

### 5.4 Hue

**OpenCV**

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180
result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let shifted = Sowilo.adjust_hue 0.1 img   (* delta in [-0.5, 0.5] *)
```

Sowilo uses `[-0.5, 0.5]` for a full hue rotation. OpenCV uses `[0, 180]` degrees for uint8 HSV.

### 5.5 Gamma correction

**OpenCV**

```python
gamma = 2.2
corrected = np.power(img_f, gamma)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let corrected = Sowilo.adjust_gamma 2.2 img
```

### 5.6 Invert

**OpenCV**

```python
inverted = 255 - img          # uint8
inverted = 1.0 - img_f       # float
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let inverted = Sowilo.invert img
```

---

## 6. Geometric Transforms

### 6.1 Resize

**OpenCV**

```python
resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
resized_nn = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let resized    = Sowilo.resize ~height:224 ~width:224 img
let resized_nn = Sowilo.resize ~interpolation:Nearest ~height:224 ~width:224 img
```

Differences:

* OpenCV takes `(width, height)`. Sowilo takes `~height` and `~width` as labeled arguments.
* Sowilo supports `Nearest` and `Bilinear` (default). OpenCV has many more modes (cubic, Lanczos, area).
* `resize` works on any dtype. For bilinear interpolation it casts to float32 internally.

### 6.2 Crop

**OpenCV**

```python
cropped = img[y:y+h, x:x+w]
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let cropped = Sowilo.crop ~y:10 ~x:20 ~height:100 ~width:100 img
let centered = Sowilo.center_crop ~height:224 ~width:224 img
```

`center_crop` computes the offset automatically. OpenCV has no built-in center crop.

### 6.3 Flip

**OpenCV**

```python
flipped_h = cv2.flip(img, 1)    # horizontal
flipped_v = cv2.flip(img, 0)    # vertical
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let flipped_h = Sowilo.hflip img
let flipped_v = Sowilo.vflip img
```

### 6.4 Rotate

**OpenCV**

```python
rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotated_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let rotated    = Sowilo.rotate90 img            (* 90 degrees counter-clockwise *)
let rotated_cw = Sowilo.rotate90 ~k:(-1) img    (* 90 degrees clockwise *)
let rotated_180 = Sowilo.rotate90 ~k:2 img      (* 180 degrees *)
```

`rotate90` only handles multiples of 90 degrees. OpenCV's `cv2.getRotationMatrix2D` + `cv2.warpAffine` for arbitrary angles has no equivalent.

### 6.5 Pad

**OpenCV**

```python
padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=0)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let padded = Sowilo.pad (10, 10, 20, 20) img            (* zero-padded *)
let padded = Sowilo.pad ~value:0.5 (10, 10, 20, 20) img (* custom fill *)
```

Sowilo supports constant padding only. OpenCV also has reflect, replicate, and wrap modes.

---

## 7. Spatial Filtering

### 7.1 Gaussian blur

**OpenCV**

```python
blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
blurred = cv2.GaussianBlur(img, (7, 7), sigmaX=1.5)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let blurred = Sowilo.gaussian_blur ~sigma:1.5 img
let blurred = Sowilo.gaussian_blur ~sigma:1.5 ~ksize:7 img
```

Sowilo defaults `ksize` to `2 * ceil(3 * sigma) + 1`, which captures 99.7% of the distribution. OpenCV lets you pass `(0, 0)` for automatic sizing. Sowilo uses separable convolution internally, same as OpenCV.

### 7.2 Box blur (averaging)

**OpenCV**

```python
blurred = cv2.blur(img, (5, 5))
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let blurred = Sowilo.box_blur ~ksize:5 img
```

Sowilo uses a square kernel. OpenCV's `cv2.blur` supports rectangular kernels.

### 7.3 Median blur

**OpenCV**

```python
blurred = cv2.medianBlur(img, 5)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let blurred = Sowilo.median_blur ~ksize:5 img
```

`ksize` must be a positive odd integer. Note: `median_blur` is **not differentiable** -- gradient is zero almost everywhere.

### 7.4 Custom kernels (filter2d)

**OpenCV**

```python
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float32)
edges = cv2.filter2D(img, -1, kernel)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let kernel = Nx.create Nx.float32 [|3; 3|]
  [|-1.; -1.; -1.;
    -1.;  8.; -1.;
    -1.; -1.; -1.|]

let edges = Sowilo.filter2d kernel img
```

Both apply 2D convolution with same-size padding. Note the argument order: Sowilo takes `kernel` first, then `img`. OpenCV takes `src`, `ddepth`, `kernel`.

### 7.5 Sharpening (unsharp mask)

**OpenCV**

```python
blurred = cv2.GaussianBlur(img, (0, 0), sigma)
sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let sharpened = Sowilo.unsharp_mask ~sigma:1.0 img
let sharpened = Sowilo.unsharp_mask ~sigma:1.0 ~amount:1.5 img
```

`amount` defaults to `1.0`. The formula is `img + amount * (img - gaussian_blur ~sigma img)`.

---

## 8. Morphological Operations

### 8.1 Structuring elements

**OpenCV**

```python
kernel_rect    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_cross   = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let kernel_rect    = Sowilo.structuring_element Rect (5, 5)
let kernel_cross   = Sowilo.structuring_element Cross (5, 5)
let kernel_ellipse = Sowilo.structuring_element Ellipse (5, 5)
```

Both produce a binary mask. Dimensions must be positive odd integers.

### 8.2 Erode and dilate

**OpenCV**

```python
eroded  = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let eroded  = Sowilo.erode ~kernel img
let dilated = Sowilo.dilate ~kernel img
```

Sowilo does not have an `iterations` parameter. Apply the operation multiple times if needed.

### 8.3 Compound operations

**OpenCV**

```python
opened   = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed   = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let opened   = Sowilo.opening ~kernel img
let closed   = Sowilo.closing ~kernel img
let gradient = Sowilo.morphological_gradient ~kernel img
```

* `opening` = erode then dilate (removes small bright regions).
* `closing` = dilate then erode (fills small dark regions).
* `morphological_gradient` = dilate - erode (highlights edges).

OpenCV also has `MORPH_TOPHAT`, `MORPH_BLACKHAT`, and `MORPH_HITMISS`. Sowilo does not provide these.

---

## 9. Edge Detection

### 9.1 Sobel

**OpenCV**

```python
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let gx, gy = Sowilo.sobel gray               (* ksize defaults to 3 *)
let gx, gy = Sowilo.sobel ~ksize:5 gray
```

Sowilo returns both gradients as a tuple. OpenCV requires two calls. Input must have `C = 1`.

### 9.2 Scharr

**OpenCV**

```python
gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let gx, gy = Sowilo.scharr gray
```

Scharr is more rotationally accurate than Sobel with `ksize=3`.

### 9.3 Laplacian

**OpenCV**

```python
laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let laplacian = Sowilo.laplacian gray
let laplacian = Sowilo.laplacian ~ksize:5 gray
```

### 9.4 Canny

**OpenCV**

```python
edges = cv2.Canny(gray_u8, 100, 200)
```

**Sowilo**

<!-- $MDX skip -->
```ocaml
let edges = Sowilo.canny ~low:0.1 ~high:0.2 gray
let edges = Sowilo.canny ~low:0.1 ~high:0.2 ~sigma:2.0 gray
```

Differences:

* OpenCV takes integer thresholds on uint8 pixel values. Sowilo takes float thresholds on `[0, 1]` values.
* Sowilo includes a built-in Gaussian blur controlled by `~sigma` (defaults to `1.4`). OpenCV expects you to blur beforehand.
* `canny` returns `1.0` for edge pixels, `0.0` for non-edges.
* **Not differentiable**: uses non-maximum suppression and hysteresis thresholding.

---

## 10. Differentiable Pipelines

This is Sowilo's key advantage over OpenCV. Because operations are expressed as Nx tensor computations, they compose with `Rune.grad` and `Rune.vmap`.

### 10.1 Gradient through image processing

No OpenCV equivalent exists. OpenCV operations are opaque C++ -- you cannot backpropagate through them.

<!-- $MDX skip -->
```ocaml
(* Compute the gradient of a loss through an image processing pipeline *)
let pipeline params img =
  img
  |> Sowilo.adjust_brightness params.brightness
  |> Sowilo.adjust_contrast params.contrast
  |> Sowilo.gaussian_blur ~sigma:params.sigma

(* Differentiate the loss w.r.t. a parameter *)
let loss_fn brightness img target =
  let processed = img |> Sowilo.adjust_brightness brightness in
  let diff = Nx.sub processed target in
  Nx.sum (Nx.mul diff diff)

let grad_fn = Rune.grad loss_fn
let grad_brightness = grad_fn 1.2 img target
```

This works because `adjust_brightness`, `adjust_contrast`, `gaussian_blur`, and most other Sowilo operations are built from differentiable Nx primitives.

### 10.2 Batch processing with vmap

**OpenCV**

```python
# Manual loop over batch
results = [cv2.GaussianBlur(img, (0, 0), 1.5) for img in batch]
```

**Sowilo** with `Rune.vmap`:

<!-- $MDX skip -->
```ocaml
(* Apply Gaussian blur to a batch of images in one call *)
let blur_batch = Rune.vmap (Sowilo.gaussian_blur ~sigma:1.5)
let blurred_batch = blur_batch batch   (* batch shape: [N; H; W; C] *)
```

### 10.3 What is differentiable?

| Operation                                                         | Differentiable                                |
| ----------------------------------------------------------------- | --------------------------------------------- |
| `to_float`, `to_uint8`                                            | Yes                                           |
| `normalize`, `threshold`                                          | Yes                                           |
| `to_grayscale`                                                    | Yes                                           |
| `rgb_to_hsv`, `hsv_to_rgb`                                        | Yes                                           |
| `adjust_brightness/contrast/saturation/hue/gamma`                 | Yes                                           |
| `invert`                                                          | Yes                                           |
| `resize` (bilinear)                                               | Yes                                           |
| `crop`, `center_crop`                                             | Yes                                           |
| `hflip`, `vflip`, `rotate90`                                      | Yes                                           |
| `pad`                                                             | Yes                                           |
| `gaussian_blur`, `box_blur`                                       | Yes                                           |
| `filter2d`, `unsharp_mask`                                        | Yes                                           |
| `erode`, `dilate`, `opening`, `closing`, `morphological_gradient` | Yes                                           |
| `sobel`, `scharr`, `laplacian`                                    | Yes                                           |
| `median_blur`                                                     | **No** (sort-based, gradient is zero)         |
| `canny`                                                           | **No** (non-maximum suppression + hysteresis) |

---

## 11. What Sowilo Doesn't Have

Sowilo is a focused library for differentiable image processing primitives. It does not cover:

* **Image I/O** -- no `imread`, `imwrite`. Use an external library to load/save images as `Nx.t` tensors.
* **Video** -- no `VideoCapture`, `VideoWriter`, or frame-by-frame processing.
* **GUI** -- no `imshow`, `waitKey`, or window management.
* **Drawing** -- no `rectangle`, `circle`, `putText`, or shape rendering.
* **Feature detection** -- no SIFT, ORB, AKAZE, or keypoint matching.
* **Contour detection** -- no `findContours`, `drawContours`, or shape analysis.
* **Object detection** -- no Haar cascades, HOG detectors, or DNN module.
* **Camera calibration** -- no `calibrateCamera`, `undistort`, or stereo vision.
* **Arbitrary affine/perspective transforms** -- no `warpAffine`, `warpPerspective`, or rotation by arbitrary angles.
* **Additional color spaces** -- no Lab, YUV, or Bayer conversions.
* **Adaptive thresholding** -- no `adaptiveThreshold` or Otsu's method.
* **Histogram operations** -- no `calcHist`, `equalizeHist`, or CLAHE.
* **Additional border modes** -- only constant padding (no reflect, replicate, or wrap).
* **Connected components** -- no `connectedComponents` or label analysis.

If you need these, use OpenCV from Python or a dedicated OCaml binding. Sowilo focuses on the subset of operations useful in differentiable ML pipelines.

---

## 12. Quick Cheat Sheet

| Task                   | OpenCV                                          | Sowilo                                                       |
| ---------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Load image             | `cv2.imread("f.jpg")`                           | N/A (use external I/O)                                       |
| uint8 to float         | `img.astype(np.float32) / 255.0`                | `Sowilo.to_float img`                                        |
| float to uint8         | `(img * 255).clip(0,255).astype(np.uint8)`      | `Sowilo.to_uint8 img`                                        |
| Normalize              | `(img - mean) / std`                            | `Sowilo.normalize ~mean ~std img`                            |
| Threshold              | `cv2.threshold(img, t, 1.0, THRESH_BINARY)`     | `Sowilo.threshold t img`                                     |
| To grayscale           | `cv2.cvtColor(img, COLOR_BGR2GRAY)`             | `Sowilo.to_grayscale img`                                    |
| RGB to HSV             | `cv2.cvtColor(img, COLOR_RGB2HSV)`              | `Sowilo.rgb_to_hsv img`                                      |
| HSV to RGB             | `cv2.cvtColor(img, COLOR_HSV2RGB)`              | `Sowilo.hsv_to_rgb img`                                      |
| Brightness             | `np.clip(img * f, 0, 1)`                        | `Sowilo.adjust_brightness f img`                             |
| Contrast               | manual per-channel math                         | `Sowilo.adjust_contrast f img`                               |
| Saturation             | manual HSV manipulation                         | `Sowilo.adjust_saturation f img`                             |
| Hue shift              | manual HSV manipulation                         | `Sowilo.adjust_hue delta img`                                |
| Gamma                  | `np.power(img, gamma)`                          | `Sowilo.adjust_gamma gamma img`                              |
| Invert                 | `1.0 - img`                                     | `Sowilo.invert img`                                          |
| Resize                 | `cv2.resize(img, (w, h))`                       | `Sowilo.resize ~height:h ~width:w img`                       |
| Crop                   | `img[y:y+h, x:x+w]`                             | `Sowilo.crop ~y ~x ~height:h ~width:w img`                   |
| Center crop            | manual offset computation                       | `Sowilo.center_crop ~height:h ~width:w img`                  |
| Horizontal flip        | `cv2.flip(img, 1)`                              | `Sowilo.hflip img`                                           |
| Vertical flip          | `cv2.flip(img, 0)`                              | `Sowilo.vflip img`                                           |
| Rotate 90              | `cv2.rotate(img, ROTATE_90_CCW)`                | `Sowilo.rotate90 img`                                        |
| Pad                    | `cv2.copyMakeBorder(img, t, b, l, r, ...)`      | `Sowilo.pad (t, b, l, r) img`                                |
| Gaussian blur          | `cv2.GaussianBlur(img, (0,0), sigma)`           | `Sowilo.gaussian_blur ~sigma img`                            |
| Box blur               | `cv2.blur(img, (k, k))`                         | `Sowilo.box_blur ~ksize:k img`                               |
| Median blur            | `cv2.medianBlur(img, k)`                        | `Sowilo.median_blur ~ksize:k img`                            |
| Custom kernel          | `cv2.filter2D(img, -1, kernel)`                 | `Sowilo.filter2d kernel img`                                 |
| Sharpen                | manual unsharp mask                             | `Sowilo.unsharp_mask ~sigma img`                             |
| Structuring element    | `cv2.getStructuringElement(shape, (h, w))`      | `Sowilo.structuring_element shape (h, w)`                    |
| Erode                  | `cv2.erode(img, kernel)`                        | `Sowilo.erode ~kernel img`                                   |
| Dilate                 | `cv2.dilate(img, kernel)`                       | `Sowilo.dilate ~kernel img`                                  |
| Opening                | `cv2.morphologyEx(img, MORPH_OPEN, kernel)`     | `Sowilo.opening ~kernel img`                                 |
| Closing                | `cv2.morphologyEx(img, MORPH_CLOSE, kernel)`    | `Sowilo.closing ~kernel img`                                 |
| Morphological gradient | `cv2.morphologyEx(img, MORPH_GRADIENT, kernel)` | `Sowilo.morphological_gradient ~kernel img`                  |
| Sobel                  | `cv2.Sobel(img, CV_32F, dx, dy)`                | `Sowilo.sobel img` (returns `(gx, gy)`)                      |
| Scharr                 | `cv2.Scharr(img, CV_32F, dx, dy)`               | `Sowilo.scharr img` (returns `(gx, gy)`)                     |
| Laplacian              | `cv2.Laplacian(img, CV_32F)`                    | `Sowilo.laplacian img`                                       |
| Canny                  | `cv2.Canny(img, low, high)`                     | `Sowilo.canny ~low ~high img`                                |
| Backprop through ops   | not possible                                    | `Rune.grad f` works on all ops except `median_blur`, `canny` |
| Batch processing       | manual loop                                     | `Rune.vmap f` over batch dimension                           |
