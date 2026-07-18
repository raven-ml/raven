# Nx C backend

This directory contains `nx.c`, the default implementation of the
`nx.backend` virtual library. It is self-contained C11 on every supported
platform. On macOS, eligible floating-point matrix multiplications are routed
automatically to the system Accelerate framework; all other operations use the
owned kernels in this directory.

There is no runtime backend selection or external BLAS/LAPACK configuration.
The owned GEMM is both the non-macOS implementation and the fallback for small,
strided, low-precision, integer, or otherwise ineligible products.

## Representation and ABI

The OCaml tensor record and `nx_c_ndarray` agree on four fields in this order:
buffer, shape, strides, and offset. Shapes, strides, and offsets are expressed
in logical elements. Packed 4-bit dtypes are the only exception at the storage
boundary. `test/test_backend_c.ml` pins the record layout and the mapping from
`Nx_buffer` kinds to C dtype tags.

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
ineligible call falls through to the owned path. The backend-local correctness
suite compares automatic Accelerate results with the owned oracle; the
backend-local benchmark forces owned GEMM so fallback performance remains
visible on Accelerate machines.

## FFT and linear algebra

FFT uses owned mixed-radix Cooley-Tukey kernels for factors through 13, with
Bluestein for lengths containing larger prime factors. Plans are cached behind
the engine's thread-safe plan cache. Forward transforms are unnormalized;
frontend norm handling remains in Nx.

Cholesky, triangular solve, QR, symmetric/Hermitian eigendecomposition, general
eigendecomposition, and SVD are implemented in the `tri`, `qr`, `eigh`, `eig`,
and `svd` translation units. Public correctness is expressed as reconstruction,
residual, orthogonality, dtype, layout, and batching properties in the Nx backend
contract. Backend-local tests retain only forced algorithm/workspace and ABI
invariants that the public interface cannot express.

## Maintenance

Fast correctness belongs to the normal `@runtest` alias. Generic backend
semantics live in `packages/nx/test/backend_contract.ml`; tests here cover C ABI,
engine, storage conversion, internal dispatch, workspace, aliasing, threading,
and forced routing. Longer linalg and eigensolver fixtures plus scaled threaded
fold stress run under the single
`@packages/nx/lib/backend_c/test/backend-stress` alias and are included in CI.

Public performance belongs to `packages/nx/bench`, including matmul, FFT, and
linalg suites. `bench/bench_owned_gemm.ml` is deliberately the sole local
benchmark because the normal macOS path would otherwise hide fallback
regressions behind Accelerate.
