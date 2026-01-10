# Changelog

All notable changes to this project will be documented in this file.

- Only document user-facing changes (features, bug fixes, performance improvements, API changes, etc.)
- Add new entries at the top of the appropriate section (most recent first)

## [1.0.0~alpha3] - Unreleased

### Nx

- Add missing bool→low-precision cast support (f16/bf16/fp8) in the C backend. (@tmattio)
- Add UInt32/UInt64 dtypes, rename complex dtypes to Complex64/Complex128, and drop Complex16/QInt8/QUInt8/Int/NativeInt as tensor element dtypes. (@tmattio)
- Move the `Rng` module from Rune into Nx. All random number generation functions (`rand`, `randn`, `randint`, etc.) now require an explicit `~key:Rng.key` parameter for reproducibility and stateless RNG. Use `Rng.key seed` to create a key and `Rng.split` to derive independent subkeys. (@tmattio)
- Add instrumentation support with `Nx_core.Instrumentation` module for tracing and profiling Nx operations. All frontend functions now emit spans via configurable hooks. (@tmattio)
- Add optional `?out` parameter to every relevant Nx function to allow users to provide pre-allocated output buffers. (@tmattio)
- Fix critical correctness issue in fancy slicing (`L`) where permutations were ignored if the number of indices matched the dimension size (e.g., `slice [L [1; 0]] x` returned `x` unmodified). (@tmattio)
- Rewrite `slice` implementation to use `as_strided` for contiguous operations, reducing overhead to **O(1)** for view-based slices and separating gather operations for better performance. (@tmattio)
- Optimize `set_slice` by replacing scalar-loop index calculations with vectorized coordinate arithmetic, significantly improving performance for fancy index assignments. (@tmattio)
- Improve `einsum` performance **8–20×** with greedy contraction path optimizer (e.g., MatMul 100×100 f32 207.83 µs → 10.76 µs, **19×**; BatchMatMul 200×200 f32 8.78 ms → 435.39 µs, **20×**) (@tmattio)
- Rewrite `diagonal` using flatten + gather approach instead of O(N²) eye matrix masking, reducing memory from O(N²) to O(N) (@tmattio)
- Add `?out` parameter to `matmul` allowing users to provide a pre-allocated output buffer, reducing allocation overhead by ~1μs per call for small matrices and enabling buffer reuse in tight loops (@tmattio)

### Rune

- Rewrite `Autodiff` module to fix critical JVP correctness issues, enable higher-order derivatives (nested gradients), and introduce `vjp` as a first-class primitive. (@tmattio)
- Add autodiff support for `as_strided`, enabling gradients through slicing and indexing operations (@tmattio)
- Add autodiff support for `cummax` and `cummin` cumulative operations (@tmattio)
- Add autodiff support for FFT operations (@tmattio)
- Add autodiff support for some linear algebra operations: QR decomposition (`qr`), Cholesky decomposition (`cholesky`), and triangular solve (`triangular_solve`). (@tmattio)

### Kaun

- Reinitialize dataset each epoch to avoid iterator exhaustion (#147, @Shocker444, @tmattio)

### Quill

- Add support for 32 bit systems by replacing ocaml-crunch with a `quill-crunch` chunked asset bundler in release builds to keep generated modules under 32-bit string limits. (@tmattio)

## [1.0.0~alpha2] - 2025-11-03

We're excited to announce the release of Raven 1.0.0~alpha2! Less than a month after alpha1, this release notably includes contributions from Outreachy applicants in preparation for the upcoming _two_ internships.

Some highlights from this release include:

- NumPy-compatible text I/O with `Nx_io.{save,load}_text`
- Lots of new functions in Nx/Rune, including neural-net ones `dropout`, `log_softmax`, `batch_norm`, `layer_norm`, and activation functions like `celu` and `celu`, and generic ones like `conjugate`, `index_put`, and more.
- Addition of `.top` libraries for `nx`, `rune`, and `hugin` that auto-install pretty-printers in the OCaml toplevel. You can run e.g. `#require "nx.top"`.
- Addition of a visualization API in Fehu via the new `fehu.visualize` library, supporting video recording.
- Redesign of Kaun core datastructure and checkpointing subsystem for complete snapshotting.
- Many, many bug fixes and correctness improvements.

We've also made numerous performance improvements across the board:

- Nx elementwise ops: 5–50× faster (e.g., Add 50×50 f32 88.81 µs → 1.83 µs, **48×**; Mul 100×100 f32 78.51 µs → 2.41 µs, **33×**).
- Nx conv2d: **4–5×** faster on common shapes; up to **115×** on heavy f64 batched cases (e.g., B16 C64→128 16×16 K3 f64 1.61 s → 13.96 ms).
- Rune autodiff: **1.2–3.7×** faster on core grads (e.g., MatMulGrad Medium 34.04 ms → 11.91 ms, **2.86×**; Large 190.19 ms → 50.97 ms, **3.73×**).
- Talon dataframes: big wins in joins and group-bys (Join 805.35 ms → 26.10 ms, **31×**; Group-by 170.80 ms → 19.03 ms, **9×**; Filter 9.93 ms → 3.39 ms, **3×**).
- Saga tokenizers: realistic workloads **4–17%** faster (e.g., WordPiece encode single 136.05 µs → 115.92 µs, **1.17×**; BPE batch_32 24.52 ms → 22.27 ms, **1.10×**)

We're closing 8 user-reported issues or feature requests and are totalling 30 community contributions from 8 unique contributors.

### Nx

- Fix einsum output axis ordering for free axes (e.g., `i,jk->jki`, `ij,klj->kli`) by correcting final transpose permutation and intermediate left-axis reordering. (@tmattio)
- Add `Nx_io.Cache_dir` module with consolidated cache directory utilities respecting `RAVEN_CACHE_ROOT`, `XDG_CACHE_HOME`, and `HOME` fallback, replacing project-specific cache logic across the whole raven ecosystem (#134, @Arsalaan-Alam)
- Add `Nx_io.save_txt` / `Nx_io.load_txt` with NumPy-compatible formatting, comments, and dtype support (#120, @six-shot)
- Optimize `multi_dot` for matrix chains, reducing intermediate allocations and improving performance (@tmattio)
- Add public `index_put` function for indexed updates (@tmattio)
- Clarify `reshape` documentation to match its view-only semantics (@tmattio)
- Provide `nx.top`, `rune.top`, and `hugin.top` libraries that auto-install pretty printers in the OCaml toplevel and update Quill to load them (@tmattio)
- Add `ifill` for explicit in-place fills and make `fill` return a copied tensor (@tmattio)
- Speed up contiguous elementwise ops via vectorized loops (@tmattio)
- Fast-path contiguous single-axis reductions to avoid iterator fallback (@tmattio)
- Speed up float reductions with contiguous multi-axis fast paths (@tmattio)
- Fast-path padding-free `unfold` to lower conv2d overhead (@tmattio)
- Move neural-network operations (softmax, log_softmax, relu, gelu, silu, sigmoid, tanh) from Kaun to Nx (@tmattio)
- Add public `conjugate` function for complex number conjugation (#125, @Arsalaan-Alam)
- Fix complex vdot to conjugate first tensor before multiplication, ensuring correct mathematical behavior (#123, @Arsalaan-Alam)
- Update comparison and conditional operations to use boolean tensors (#115, @nirnayroy)
- Add support for rcond parameter and underdetermined systems to `lstsq` (#102, @Shocker444)
- Fix `matrix_rank`/`pinv` Hermitian fast paths to use eigen-decomposition and match NumPy for complex inputs (#96, @six-shot, @tmattio)
- Optimize matmul BLAS dispatch for strided tensors, improving matrix multiplication performance (@tmattio)
- Fix slow builds reported since alpha1 (#88, @tmattio)
- Fix macOS ARM crash when loading extended bigarray kinds (@tmattio)
- Add float16 and bfloat16 support to safetensors I/O, including precise conversions that preserve denormals/NaNs (#84, @six-shot, @tmattio)
- Refined `View` internals for leaner contiguity checks and stride handling, cutting redundant materialization on hot paths (@tmattio)
- Merge `Lazy_view` into the core `View` API so movement ops operate on a single composed view (@tmattio)
- Documented the reworked `View` interface (@tmattio)
- Documented the `Symbolic_shape` interface (@tmattio)
- Added Accelerate framework flag when compiling on macOS, fixing issues in some environments (#129, @nirnayroy)

### Hugin

- Fix random `SIGBUS`/bus errors on macOS when closing `Hugin.show` windows by
  destroying SDL windows with the correct pointer in the finalizer. (@tmattio)
- Let `Hugin.show` windows close cleanly via the window button or `Esc`/`q`, avoiding frozen macOS REPL sessions (@tmattio)

### Rune

- Add `Rune.no_grad` and `Rune.detach` to mirror JAX stop-gradient semantics (@tmattio)
- Improve gradient performance slightly by replace the reverse-mode tape's linear PhysicalTbl with an identity hash table (@tmattio)
- Fix `Rune.Rng.shuffle` flattening outputs for multi-dimensional tensors; the
  shuffle now gathers along axis 0 and keeps shapes intact (@tmattio)
- Replace `Rune.Rng.truncated_normal` clipping with rejection sampling so
  samples stay inside the requested interval without boundary spikes (@tmattio)
- Add support for categorical sampling with `Rune.Rng.categorical` (#89, @nirnayroy)
- Allow plain `llvm-config` in discovery, fixing build in some platforms (#71, @stepbrobd)

### Kaun

- Added Similarity and Polysemy analysis to the BERT example (#137, @nirnayroy)
- Support attention masks via the new `Kaun.Attention` module (@tmattio)
- Support loading sharded Hugging Face safetensors (@tmattio)
- Fix BERT and GPT‑2 model loading (@tmattio)
- API simplification: removed type parameters from public types; `Ptree` now supports mixed‑dtype trees via packed tensors with typed getters. (@tmattio)
- Checkpointing overhaul: versioned `Train_state` with schema tagging, explicit `Checkpoint.{Snapshot,Artifact,Manifest,Repository}` (retention, tags, metadata), and simple save/load helpers for snapshots and params. (@tmattio)
- Overhaul dataset combinators: derive tensor specs from Rune dtype, fix sampling/window bugs, validate weighted sampling, and respect `drop_remainder` (@tmattio)
- Make dataset `prefetch` truly asynchronous with background domains and allow reusing an external Domainslib pool via `parallel_map ~pool` (@tmattio)
- Use `Dataset.iter` for epoch batches to reduce overhead (@tmattio)
- Update BERT and GPT-2 tokenizer cache to use `Nx.Cache` for consistent cache directory resolution (#134, @Arsalaan-Alam)
- Honor text dataset encodings via incremental Uutf decoding (#122, @Satarupa22-SD).
- Preserve empty sequential modules when unflattening so indices stay aligned for checkpoint round-tripping (@tmattio)
- Prevent `Training.fit`/`evaluate` from consuming entire datasets eagerly and fail fast when a dataset yields no batches, avoiding hangs and division-by-zero crashes (@tmattio)
- Allow metric history to tolerate metrics that appear or disappear between epochs so dynamic metric sets no longer raise during training (@tmattio)
- Make `Optimizer.clip_by_global_norm` robust to zero gradients and empty parameter trees to avoid NaNs during training (@tmattio)
- Split CSV loader into `from_csv` and `from_csv_with_labels` to retain labels when requested (#114, @Satarupa22-SD)
- Implement AUC-ROC and AUC-PR in Kaun metrics and simplify their signatures (#124, #131, @Shocker444)
- Add mean absolute percentage error, explained variance, R² (with optional adjustment), KL-divergence, and top-k accuracy to Kaun metrics (@tmattio)
- Add NDCG, MAP, and MRR ranking metrics to Kaun metrics (@tmattio)
- Add BLEU, ROUGE, and METEOR metrics to Kaun for pre-tokenized sequences, removing tokenizer dependencies (@tmattio)
- Add SSIM, IoU, and Dice metrics for vision workloads in Kaun (@tmattio)

### Talon

- Remove automatic sentinel-based null detection for numeric columns; explicit masks (via [_opt] constructors) now define missing data semantics (@tmattio)
- Replace join nested loops with hashed join indices, cutting lookup from O(n·m) to near O(n) (@tmattio)
- Reuse a shared Nx-based column reindexer so filter/sample paths avoid repeated array copies (@tmattio)
- Fix `fillna` to honor column null masks and replacements, restoring expected nullable semantics (@tmattio)
- Preserve null masks when reindexing during joins so sentinel values remain valid data (@tmattio)
- Handle numeric index columns in `pivot`, preventing distinct keys from collapsing into a single bucket (@tmattio)
- Respect null masks when serializing numeric columns to JSON, emitting JSON `null` instead of sentinel values (@tmattio)
- Detect big integers as int64 in Talon CSV loader (#121, @Arsalaan-Alam)
- Allow forcing column types in Talon JSON loader (#104, @nirnayroy)
- Add documentation to compare Talon and Pandas (#154, Satarupa22-SD)

### Saga

- Remove legacy `Normalizers.nmt` and `Normalizers.precompiled` constructors (and their JSON serializers) so the public surface only advertises supported normalizers (@tmattio)
- Tighten template processor JSON parsing: require integer type ids, drop the legacy special-token list format, and ensure multi-id special tokens round-trip with the new record fields (@tmattio)
- Make tokenizer JSON loading tolerant of HuggingFace quirks (missing `model.type`, string-encoded merges), restoring compatibility with upstream `tokenizer.json` files (@tmattio)
- Cache byte-level encode/decode lookup tables to avoid rebuilding them during tokenization, trimming avoidable allocations (@tmattio)
- Skip BPE dropout sampling when dropout is disabled, removing redundant RNG work on common hot paths (@tmattio)
- Fix Unigram tokenization so longest matches are emitted without aborting the sequence when a vocab hit occurs (@tmattio)
- Recompute pad token ids when the pad special string changes, preventing padding with stale ids (@tmattio)
- Fix Unigram `token_to_id`/`id_to_token` vocabulary lookups (#117, @RidwanAdebosin)
- Optimize `Pre_tokenizers.whitespace` to reduce allocations and improve tokenization performance (@tmattio)
- Simplify tokenizers interface (@tmattio)

### Sowilo

- Add `resize` (nearest & bilinear) that works for 2D, batched, and NHWC tensors (@tmattio)
- Update grayscale conversion and RGB/BGR channel swaps to run entirely on Rune ops, keeping batched inputs compatible with JIT backends (@tmattio)
- Make `median_blur` compute the true median so salt-and-pepper noise is removed as expected (@tmattio)
- Fix `erode`/`dilate` so custom structuring elements (e.g. cross vs. square) and batched tensors produce the correct morphology result (@tmattio)

### Fehu

- Added snapshot-based save/load for DQN and REINFORCE agents (#127, @RidwanAdebosin, @tmattio)
- Added typed `Render` payloads with enforced `render_mode` selection in `Env.create`, auto human-mode rendering, and vectorized `Env.render` accessors so environments consistently expose frames for downstream tooling (@tmattio)
- Introduced the `Fehu_visualize` library with ffmpeg/gif/W&B sinks, overlay combinators, rollout/evaluation recorders, and video wrappers for single and vectorized environments, providing a cohesive visualization stack for Fehu (@tmattio)
- Added a `Fehu.Policy` helper module (random/deterministic/greedy) and sink `with_*` guards so visualization sinks handle directory creation and cleanup automatically (@tmattio)
- Added `Buffer.Replay.sample_tensors` to streamline batched training loops and exploration handling (@tmattio)
- Reworked `Fehu_algorithms.Dqn` around `init`/`step`/`train` primitives with functional state, warmup control, and snapshotting helpers (@tmattio)
- Rebuilt `Fehu_algorithms.Reinforce` on the same `init`/`step`/`train` interface with optional baselines, tensor-based rollouts, snapshot save/load, and updated tests/examples/docs using the new workflow (@tmattio)
- Upgraded the GridWorld environment to return ANSI and RGB-array frames using the new render types, and updated the DQN example to optionally record pre- and post-training rollouts via `FEHU_DQN_RECORD_DIR` using `Fehu_visualize` sinks (@tmattio)
- Reworked space sampling to return `(value, next_rng)` and split keys internally, fixing correlated draws in Box/Multi-discrete/Tuple/Dict/Sequence/Text samplers while adding `Space.boundary_values` for deterministic compatibility checks (@tmattio)
- Extended vectorized environments to reuse space boundary probes and now store structured `final_observation` payloads in `Info`, improving downstream consumption (@tmattio)
- Added `Buffer.Replay.add_many` and `Buffer.Replay.sample_arrays`, preserved backing storage on `clear`, and exposed struct-of-arrays batches for vectorised learners (@tmattio)
- Tightened `Env.create` diagnostics with contextual error messages and an optional `~validate_transition` hook for custom invariants (@tmattio)
- Enriched `Wrapper` utilities with `map_info`, Box `clip_action`/`clip_observation`, and time-limit info reporting elapsed steps (@tmattio)
- Upgraded `Info` values to carry int/float/bool arrays with stable JSON round-tripping (handling NaN/∞) and sorted metadata serialization for deterministic diffs (@tmattio)
- Improved training helpers: Welford-based normalization with optional unbiased variance, documented `done = terminated || truncated`, and returned `nan` when explained variance is undefined (@tmattio)
- Treat time-limit truncations as terminals when computing rollout advantages and expose the `truncated` flag in buffer steps (@tmattio)
- Require callers of `Training.compute_gae` to pass final bootstrapping values and ensure `Training.evaluate` feeds the current observation to policies (@tmattio)
- Allow `Space.Sequence.create` to omit `max_length`, keeping sequences unbounded above while preserving validation and sampling semantics (@tmattio)
- Validate vectorized environments by round-tripping sample actions/observations across every instance, preventing incompatible spaces from slipping through (@tmattio)
- Finish clipped value loss support in Fehu.Training (#119, @nirnayroy)

### Nx-datasets

- Migrate to `Nx.Cache` for cache directory resolution, enabling consistent behavior. (#133, @Arsalaan-Alam)
- Fix cache directory resolution to respect `RAVEN_CACHE_ROOT` (or fall back to `XDG_CACHE_HOME`/`HOME`), allowing custom cache locations. (#128, @Arsalaan-Alam)
- Switch CIFAR-10 loader to the binary archive so parsing succeeds again (@tmattio)
- Add a CIFAR-10 example (@tmattio)
- Standardize dataset examples on `Logs` (@tmattio)
- Use `Logs` for dataset loader logging (#95, @Satarupa22-SD)

## [1.0.0~alpha1] - 2025-10-02

This release expands the Raven ecosystem with three new libraries (Talon, Saga, Fehu) and significant enhancements to existing ones. `alpha1` focuses on breadth—adding foundational capabilities across data processing, NLP, and reinforcement learning—while continuing to iterate on core infrastructure.

### New Libraries

#### Talon - DataFrame Processing
We've added Talon, a new DataFrame library inspired by pandas and polars:
- Columnar data structures that support mixed types (integers, floats, strings, etc.) within a single table (aka heterogeneous datasets)
- Operations: filter rows, group by columns, join tables, compute aggregates
- Load and save data in CSV and JSON formats
- Seamless conversion to/from Nx arrays for numerical operations

#### Saga - NLP & Text Processing
Saga is a new text processing library for building language models. It provides:
- Tokenizers: Byte-pair encoding (BPE), WordPiece subword tokenization, and character-level splitting
- Text generation: Control output with temperature scaling, top-k filtering, nucleus (top-p) sampling, and custom sampling strategies
- Language models: Train and generate text with statistical n-gram models (bigrams, trigrams, etc.)
- I/O: Read large text files line-by-line and batch-process corpora

#### Fehu - Reinforcement Learning
Fehu brings reinforcement learning to Raven, with an API inspired by Gymnasium and Stable-Baselines3:
- Standard RL environment interface (reset, step, render) with example environments like Random Walk and CartPole
- Environment wrappers to modify observations, rewards, or episode termination conditions
- Vectorized environments to collect experience from multiple parallel rollouts
- Training utilities: Generalized advantage estimation (GAE), trajectory collection and management
- RL algorithms: Policy gradient method (REINFORCE), deep Q-learning (DQN) with replay buffer
- Use Kaun neural networks as function approximators for policies and value functions

### Major Enhancements

#### Nx - Array Computing
We've significantly expanded Nx's following early user feedback from alpha0:
- Complete linear algebra suite: LAPACK-backed operations matching NumPy including singular value decomposition (SVD), QR factorization, Cholesky decomposition, eigenvalue/eigenvector computation, matrix inverse, and solving linear systems
- FFT operations: Fast Fourier transforms (FFT/IFFT) for frequency domain analysis and signal processing
- Advanced operations: Einstein summation notation (`einsum`) for complex tensor operations, extract/construct diagonal matrices (`diag`), cumulative sums and products along axes
- Extended dtypes: Machine learning-focused types including bfloat16 (brain floating point), complex16, and float8 for reduced-precision training
- Symbolic shapes: Internal infrastructure for symbolic shape inference to enable dynamic shapes in future releases (not yet exposed in public API)
- Lazy views: Array views only copy and reorder memory when stride patterns require it, avoiding unnecessary allocations

#### Rune - Autodiff & JIT
We've continued iterating on Rune's autodiff capabilities, and made progress on upcoming features:
- Forward-mode AD: Compute Jacobian-vector products (`jvp`) for forward-mode automatic differentiation, complementing existing reverse-mode
- JIT: Ongoing development of LLVM-based just-in-time compilation for Rune computations (currently in prototype stage)
- vmap: Experimental support for vectorized mapping to automatically batch operations (work-in-progress, not yet stable)
- LLVM backend: Added compilation backend with support for LLVM versions 19, 20, and 21
- Metal backend: Continued work on GPU acceleration for macOS using Metal compute shaders

#### Kaun - Deep Learning
We've expanded Kaun with high-level APIs for deep learning. These APIs are inspired by popular Python frameworks like TensorFlow, PyTorch, and Flax, and should feel familiar to users building models in Python:
- High-level training: Keras-style `fit()` function to train models with automatic batching, gradient computation, and parameter updates
- Training state: Encapsulated training state (TrainState) holding parameters, optimizer state, and step count; automatic history tracking of loss and metrics
- Checkpoints: Save and load model weights to disk for model persistence and transfer learning
- Metrics: Automatic metric computation during training including accuracy, precision, recall, F1 score, mean absolute error (MAE), and mean squared error (MSE)
- Data pipeline: Composable dataset operations (map, filter, batch, shuffle, cache) inspired by TensorFlow's `tf.data` for building input pipelines
- Model zoo: Reference implementations of classic and modern architectures (LeNet5 for basic CNNs, BERT for masked language modeling, GPT2 for autoregressive generation) including reusable transformer components
- Ecosystem integration: Load HuggingFace model architectures (`kaun.huggingface`), access common datasets like MNIST and CIFAR-10 (`kaun.datasets`), and use standardized model definitions (`kaun.models`)

### Contributors

Thanks to everyone who contributed to this release:

- @adamchol (Adam Cholewi) - Implemented the initial `associative_scan` native backend operation for cumulative operations
- @akshay-gulab (Akshay Gulabrao)
- @dhruvmakwana (Dhruv Makwana) - Implemented `einsum` for Einstein summation notation
- @gabyfle (Gabriel Santamaria) - Built PocketFFT bindings that replaced our custom FFT kernels
- @lukstafi (Lukasz Stafiniak) - Major contributions to Fehu and FunOCaml workshop on training Sokoban agents
- @nickbetteridge
- @sidkshatriya (Sidharth Kshatriya)

## [1.0.0~alpha0] - 2025-07-05

### Initial Alpha Release

We're excited to release the zeroth alpha of Raven, an OCaml machine learning ecosystem bringing modern scientific computing to OCaml.

### Added

#### Core Libraries

- **Nx** - N-dimensional array library with NumPy-like API
  - Multi-dimensional tensors with support for several data types.
  - Zero-copy operations: slicing, reshaping, broadcasting
  - Element-wise and linear algebra operations
  - Swappable backends: Native OCaml, C, Metal
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
[1.0.0~alpha1]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha1
[1.0.0~alpha2]: https://github.com/raven-ocaml/raven/releases/tag/v1.0.0~alpha2
