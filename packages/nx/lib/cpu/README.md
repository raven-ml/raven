# Nx C backend

This directory contains `nx.cpu`, nx's kernels over host memory: the `Nx_cpu`
module, which `Nx.Backend.host` runs. It is self-contained C11 on every
supported platform. On macOS, eligible floating-point matrix multiplications are routed
automatically to the system Accelerate framework; all other operations use the
owned kernels in this directory.

There is no runtime backend selection or external BLAS/LAPACK configuration.
The owned GEMM is both the non-macOS implementation and the fallback for small,
strided, low-precision, integer, or otherwise ineligible products.

## Representation and ABI

An operand is an `Nx_array.t`, which crosses the FFI as it is: its dtype, its
view (shape, strides, offset) and its buffer, in that order, the view's fields
read at their own slots. The buffer is a host `Nx_device.Buffer.t`,
whose first byte C reads with `nx_device_buffer_host` from nx.device's
`nx_device.h`; the dtype's constructor index is its C dtype tag. Shapes,
strides, and offsets are expressed in logical elements. Packed 4-bit dtypes are
the only exception at the storage boundary. `nx_c.h`'s `_Static_assert` pins
the tag order, and every dtype and layout crosses the FFI in Nx's suites, which
would see a row of the wrong element size, class or signedness.

Kernel tables use designated initializers indexed by the dtype enum. Unsupported
dtype entries remain null and must be rejected by the common driver before a
kernel call. Integer arithmetic follows Nx's modular storage semantics;
`-fwrapv` is part of the build policy. Floating-point kernels preserve IEEE NaN
and infinity behavior and are never compiled with `-ffast-math`.

## Iteration and concurrency

`nx_c_engine.[ch]` owns validation, dimension coalescing, iterator selection,
error translation, and the shared worker pool. Kernels receive validated C data
only; worker bodies do not inspect OCaml values, allocate on the OCaml heap, or
call the runtime.

Long operations release the OCaml runtime lock in the engine funnel. The
calling thread participates in work, and pool workers operate on disjoint output
regions or explicitly partitioned scratch. New parallel paths must retain that
ownership proof and surface failures through `nx_c_status` rather than raising
from worker code. Fork handlers quiesce the pool before `fork`; the parent keeps
its workers, while the child abandons the inherited pthread state and lazily
builds a fresh pool on its first parallel operation.

## Matrix multiplication

`nx_c_matmul.c` implements direct and packed blocked GEMM across Nx dtypes. The
blocked path accumulates in the dtype's compute type and stores each output
element once. Linalg kernels use the caller-workspace entry point so pooled
factorization workers do not allocate packing buffers.

On macOS, the top-level driver uses Accelerate CBLAS only when dtype, size, and
strides are representable without copying. Accelerate runs on the calling thread
with the OCaml runtime lock released and owns its internal parallelism. Every
ineligible call falls through to the owned path. The public matmul suite checks
both routes against the sum of products; the backend-local benchmark forces
owned GEMM so fallback performance remains visible on Accelerate machines.

## FFT and linear algebra

FFT uses owned mixed-radix Cooley-Tukey kernels for factors through 13, with
Bluestein for lengths containing larger prime factors. Plans are cached behind
the engine's thread-safe plan cache. Forward transforms are unnormalized;
frontend norm handling remains in Nx.

Cholesky, triangular solve, QR, symmetric/Hermitian eigendecomposition, general
eigendecomposition, and SVD are implemented in the `tri`, `qr`, `eigh`, `eig`,
and `svd` translation units.

## Maintenance

The backend is tested through Nx's public API, in `packages/nx/test`: each
suite draws inputs on both sides of every threshold where a kernel changes
method (matmul routes, the linalg crossovers, the sort's radix path, the
engine's chunking and fork handling), at every dtype, with the large sizes
tagged `slow`. To check that the suites still reach every path, build `nx_cpu`
with `-fprofile-instr-generate -fcoverage-mapping` (and `-fprofile-instr-generate`
in `c_library_flags`), run the suites and read `llvm-cov report` over
`nx_c_*.c`.

Public performance belongs to `packages/nx/bench`, including matmul, FFT, and
linalg suites. `bench/bench_owned_gemm.ml` is deliberately the sole local
benchmark because the normal macOS path would otherwise hide fallback
regressions behind Accelerate.
