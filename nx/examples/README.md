# Nx Examples

Ten standalone examples that teach Nx from the ground up. Each builds on the
previous one, progressing from array creation to a complete data pipeline.

## Examples

| #   | Example                                                      | What You'll Learn                                                    |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------- |
| 01  | [Creating Arrays](01-creating-arrays/)                       | `zeros`, `ones`, `arange`, `linspace`, `init`, `meshgrid`, dtypes    |
| 02  | [Infix and Arithmetic](02-infix-and-arithmetic/)             | `Nx.Infix` operators (`+`, `*$`, `/`), `abs`, `sqrt`, `exp`, `clamp` |
| 03  | [Indexing and Slicing](03-indexing-and-slicing/)             | `I`, `R`, `Rs`, `A`, `.${[...]}`, `compress`, `where`, `take`        |
| 04  | [Reshaping and Broadcasting](04-reshaping-and-broadcasting/) | `reshape`, `flatten`, `transpose`, `vstack`, broadcasting rules      |
| 05  | [Reductions and Statistics](05-reductions-and-statistics/)   | `mean`, `std`, `argmax`, `cumsum`, `all`, `any`, axis parameter      |
| 06  | [Random Numbers](06-random-numbers/)                         | `Rng.key`, `Rng.split`, `Rng.uniform`, `Rng.normal`, Monte Carlo     |
| 07  | [Linear Algebra](07-linear-algebra/)                         | `@@`, `/@`, `inv`, `det`, `lstsq`, `eigh`, `svd`                     |
| 08  | [Signal Processing](08-signal-processing/)                   | `rfft`, `irfft`, `rfftfreq`, frequency filtering                     |
| 09  | [Image Processing](09-image-processing/)                     | `correlate2d`, `max_pool2d`, Sobel edges, `Nx_io.save_image`         |
| 10  | [Data Pipeline](10-data-pipeline/)                           | `Nx_datasets`, `standardize`, train/test split, nearest-centroid     |

## Running

From the repository root:

```bash
dune exec nx/examples/01-creating-arrays/main.exe
dune exec nx/examples/02-infix-and-arithmetic/main.exe
# ... and so on through 10
```

## Dependencies

- Examples 01-08 use only `nx`
- Example 09 adds `nx.io` (image I/O)
- Example 10 adds `nx.datasets` (bundled datasets)
