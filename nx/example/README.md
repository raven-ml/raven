# Nx Examples

A curated set of standalone OCaml programs demonstrating how to use the `Nx` library for everything from creating basic arrays to advanced linear algebra.

Each numbered folder is a self-contained example:

1. **01-hello-world**  
   Creating arrays: `zeros`, `ones`, `full`, ranges, identity, custom data, and `init` with a function.

2. **02-basic-operations**  
   Arithmetic (`add`, `sub`, …), indexing, in-place ops, scalar math, transforms (`transpose`, `reshape`, `flatten`), comparisons and clipping.

3. **03-broadcasting**  
   Automatic and explicit broadcasting: matrix–scalar, matrix–vector, outer sums, `broadcast_to`, and `broadcast_arrays`.

4. **04-statistics**  
   Reductions and statistical functions: `sum`, `mean`, `min`/`max`, `prod`, `var`, `std`, `argmax`/`argmin`, sorting, unique, masks (`isnan`, `isinf`), and custom `fold`.

5. **05-array-manipulation**  
   Reshaping, slicing (`get`, multi-axis), permuting axes, stacking (`vstack`, `hstack`, `dstack`, `concatenate`), splitting, expanding/squeezing dimensions, tiling/repeating, padding, flipping/rolling.

6. **06-linear-algebra**  
   Matrix ops (`matmul`, `inv`, `solve`), eigendecomposition (`eig`, `eigh`), SVD, and a simple linear regression example.

7. **07-io**  
   File I/O operations: reading/writing NPY files, NPZ archives, and image files (PNG/JPEG).

## Building & Running

From the root of the Raven repository:

```bash
dune exec nx/example/01-hello-world/hello_world.exe
dune exec nx/example/02-basic-operations/basic_operations.exe
dune exec nx/example/03-broadcasting/broadcasting.exe
dune exec nx/example/04-statistics/statistics.exe
dune exec nx/example/05-array-manipulation/array_manipulation.exe
dune exec nx/example/06-linear-algebra/linear_algebra.exe
dune exec nx/example/07-io/io_operations.exe
```

Or cd into a folder:

```bash
cd nx/example/01-plot2d
dune exec plot2d.exe
```

