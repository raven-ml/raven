# Nx Developer Guide

## Architecture

Nx is a NumPy-like N-dimensional array library with pluggable backends. The design separates high-level API from low-level execution.

### Core Components

- **[lib/core/backend_intf.ml](lib/core/backend_intf.ml)**: Backend interface defining ~50 primitive operations (tinygrad UOps inspired)
- **[lib/core/frontend.ml](lib/core/frontend.ml)**: NumPy-compatible API with broadcasting and type checking
- **[lib/backend_c/](lib/backend_c/)**: Default CPU backend using Bigarray and C stubs for performance
- **[lib/io/](lib/io/)**: I/O operations (PNG, JPEG, .npy, .npz)

Backends run inside an execution context (our C backend uses `unit`, but other
implementations often stash allocator/device state there). The functor
[`Nx_core.Make_frontend`](lib/core/nx_core.ml) threads this context through every
primitive, and the public [`lib/nx.ml`](lib/nx.ml) lazily creates the default C
context via `Nx_c.create_context`. When authoring a new backend, expose a similar
constructor so callers can manage non-trivial resources explicitly.

### Key Design Principles

1. **Backends handle execution only**: Frontend validates shapes, broadcasts arguments, casts dtypes
2. **Zero-copy views**: Slicing, reshaping, and transposing create strided views when possible
3. **NumPy compatibility**: Follow NumPy semantics for slicing, broadcasting, dimension squeezing
4. **Fail fast**: Invalid operations raise exceptions immediately

## Data Model

### Tensors

Tensors have:
- **Shape**: `int array` of dimension sizes
- **Dtype**: Type tag for element type (float32, int64, etc.)
- **Strides**: `int array` defining memory layout in elements
- **Data**: Bigarray buffer (may be shared across views)

### Strides and Memory Layout

- C-contiguous: `strides.(i) = product(shape.(i+1..))`
- Zero stride = broadcast dimension (no storage)
- Cannot reshape tensors with zero strides (would require copy)

### View Semantics

Operations that create views (no copy):
- `slice`: Extract sub-tensor via strided indexing
- `reshape`: Change shape if strides permit (contiguous or simple transposition)
- `transpose`: Permute dimensions by swapping strides
- `broadcast_to`: Add dimensions with zero strides

Operations that always copy:
- `init`, `full`, `zeros`, `ones`: Allocate contiguous storage
- Element-wise operations: Produce fresh contiguous output

### Lazy Views and Strides

Movement ops append nodes to a [`Lazy_view.t`](lib/core/lazy_view.mli) chain
instead of rewriting buffers. This keeps reshapes, transposes, and
`broadcast_to` O(1), but it also means layout queries can fail when the composed
view is too complex or still symbolic.

- `Nx.strides`/`Nx.stride` succeed only when `Lazy_view.can_get_strides` is true.
- Call `Nx.contiguous` (which forces `op_contiguous`) before inspecting layout if
  you see errors such as "view has non-materializable layout" or "view has
  symbolic shape".
- `Nx.as_strided` creates custom views; validate your stride math because the
  backend will not automatically re-check bounds.

For debugging, inspect the simplified view list via `Lazy_view.simplify` to make
sure a chain can merge into a single [`View.t`](lib/core/view.mli).

## Symbolic Shapes

Shape-polymorphic code uses [`Symbolic_shape`](lib/core/symbolic_shape.mli).

- Create symbolic dimensions with `Symbolic_shape.dynamic ~min ~max`, or pass
  `Symbolic_shape.infer` (NumPy-style -1) when reshaping.
- Bind runtime values via `Symbolic_shape.bind` before calling APIs that require
  concrete sizes (`shape`, `numel`, `strides`, etc.).
- `Symbolic_shape.resolve_reshape` fills inferred dims and returns `None` when
  element counts disagree—propagate that failure with a helpful error. Many
  frontend helpers surface the symbolic variable name through `Error.failed`.
- Delay evaluation as long as the backend accepts symbolic extents; only call
  `Symbolic_shape.eval` when a kernel genuinely needs integers.

## Development Workflow

### Building and Testing

```bash
# Build nx and dependencies
dune build nx/

# Run all tests
dune build nx/test/test_nx.exe && _build/default/nx/test/test_nx.exe

# Run specific test suite
_build/default/nx/test/test_nx.exe test "Slicing"

# Run with verbose output
_build/default/nx/test/test_nx.exe test "Broadcasting" -v
```

### Native Backend Dependencies

The C backend relies on OpenBLAS, LAPACKE, and (optionally) OpenMP. The
configurator [`lib/backend_c/config/discover.ml`](lib/backend_c/config/discover.ml)
probes these libraries via `pkg-config` and platform-specific heuristics:

- Set `PKG_CONFIG_PATH` if `openblas.pc` or `lapacke.pc` live outside default
  search paths.
- On macOS with clang, install `libomp` (e.g. `brew install libomp`) or set
  `LIBOMP_PREFIX` so headers and libraries can be found.
- When detection fails the script prints the exact compiler/linker flags; copy
  them into your environment or tweak the script before retrying.

`_OPENMP` gates parallel loops in the stubs. Benchmark both serial and parallel
builds when chasing performance regressions.

### Testing Conventions

Use Alcotest with custom test helpers:

```ocaml
(* Check tensor values with epsilon tolerance *)
check_t ~eps:1e-6 "test name" expected_shape expected_values result_tensor

(* Test edge cases *)
let test_slice_empty () =
  let x = create float32 [|2; 3|] [|1.;2.;3.;4.;5.;6.|] in
  let s = slice x [R(0, 0)] in
  check_t "empty slice" [|0; 3|] [||] s
```

Float comparisons need `~eps` parameter due to numerical precision.

### Debugging Failed Tests

1. Print intermediate shapes and values
2. Verify test expectations against NumPy behavior
3. Check that view operations don't create copies
4. Create minimal repro in `test/failing/bug_<name>.ml`

Common gotchas:
- Slice end indices are **exclusive**: `R(1, 4)` = indices 1,2,3
- Single indexing **squeezes** dimension: `get x [0]` on `[2;3]` → shape `[3]`
- `init` creates contiguous arrays; `broadcast_to` creates views with zero strides

## Adding Features

### New Operations

1. Add backend operation to [backend_intf.ml](lib/core/backend_intf.ml)
2. Implement in C backend [lib/backend_c/](lib/backend_c/)
3. Add frontend wrapper in [lib/core/frontend.ml](lib/core/frontend.ml) with broadcasting/validation
4. Write tests covering edge cases (empty arrays, broadcasting, dtypes)
5. Update [lib/nx.mli](lib/nx.mli) with documentation

**Do not** add operations unless necessary—keep the backend minimal.

### Backend Operations

Backend ops assume:
- Inputs broadcast to same shape (frontend responsibility)
- Dtypes compatible and cast (frontend responsibility)
- Shape validation done (frontend responsibility)

Backend ops simply execute the primitive operation and return a fresh tensor.

### Advanced Backend Primitives

[`lib/core/backend_intf.ml`](lib/core/backend_intf.ml) includes higher-level
primitives beyond element-wise math. When porting to a new backend, double-check
their contracts:

- **Indexing**: `op_gather`/`op_scatter` power fancy indexing. Respect
  `~mode` ("set" vs "add") and treat `~unique_indices` as an optimization hint.
- **Windowing**: `op_unfold`/`op_fold` implement im2col/col2im. Validate
  `kernel_size`, `stride`, `dilation`, and `padding` arrays before touching
  memory.
- **Transforms & Linalg**: FFT variants plus `op_qr`, `op_svd`, `op_eig`, etc.
  return fixed dtypes (e.g. SVD singular values are float64) and support batched
  inputs.
- **Custom Views**: `op_as_strided` must guarantee bounds safety. Reject
  negative or oversized strides instead of silently wrapping.

Add targeted Alcotest regressions for tricky paths, such as overlapping scatter
updates or symbolic kernel sizes.

### Data Types

Supported dtypes (see [lib/core/dtype.ml](lib/core/dtype.ml)):
- Float: `float16`, `float32`, `float64`
- Int: `int8`, `int16`, `int32`, `int64`
- Uint: `uint8`, `uint16`
- Complex: `complex32`, `complex64`

Comparison operations return `uint8` (OCaml Bigarray lacks native bool type).

## Common Pitfalls

### GADT Pattern Matching

Cannot group branches:

```ocaml
(* Wrong *)
match dtype x with
| Float32 | Float64 -> ...

(* Correct: use locally abstract types *)
let f (type a b) (x : (a, b) t) =
  match dtype x with
  | Float32 -> ...
  | Float64 -> ...
```

### Circular Dependencies

If functions call each other, use backend ops directly:

```ocaml
(* Avoid: Frontend.foo -> Frontend.bar -> Frontend.foo *)

(* Correct: Call backend ops *)
let foo x = Backend.op_add x (Backend.op_const_scalar ...)
```

### Broadcasting Semantics

NumPy-style broadcasting:
- Align shapes from right
- Dimensions match if equal or one is 1
- Missing dimensions treated as 1

Frontend handles broadcasting; backend receives same-shape inputs.

### Performance

- Use `Bigarray.unsafe_get/set` in tight loops
- Validate indices once before loop
- Batch operations to reduce backend dispatch overhead
- Prefer contiguous arrays for sequential access

### Native Constraints

- The C stubs cap tensor rank at `MAX_NDIM = 32`
  ([`lib/backend_c/nx_c_shared.h`](lib/backend_c/nx_c_shared.h)). Increase this only
  with matching OCaml/C updates.
- Packed dtypes (int4/uint4, complex16) clamp on load/store; add regression tests
  before depending on edge values.
- OpenMP is optional. Guard parallel loops with `#ifdef _OPENMP` and test serial
  builds to catch scheduling bugs.

## Code Style

- **Naming**: `snake_case` for values/functions/types
- **Modules**: `My_Module` capitalization
- **Errors**: `"function_name: error description"` format
- **Documentation**: Terse first line; explain invariants not obvious behavior
- **Type annotations**: Only when required by type checker (dtype pattern matching)

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- NumPy documentation for API compatibility reference
