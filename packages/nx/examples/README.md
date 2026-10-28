# Nx Examples

Ten standalone examples that teach Nx from the ground up. Each builds on the
previous one, progressing from array creation to machine learning.

## Examples

| #   | Example                                                      | What You'll Learn                                                    |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------- |
| 01  | [Creating Arrays](01-creating-arrays/)                       | `zeros`, `ones`, `arange`, `linspace`, `init`, `meshgrid`, dtypes    |
| 02  | [Infix and Arithmetic](02-infix-and-arithmetic/)             | Operators in `Nx.Infix.( … )` (`+`, `*$`, `/`), `abs`, `sqrt`, `clamp`     |
| 03  | [Indexing and Slicing](03-indexing-and-slicing/)             | `I`, `R`, `Rs`, `A`, `.${[...]}`, `M`, `T`, `where`              |
| 04  | [Reshaping and Broadcasting](04-reshaping-and-broadcasting/) | `reshape`, `flatten`, `transpose`, `concatenate`, broadcasting rules |
| 05  | [Reductions and Statistics](05-reductions-and-statistics/)   | `mean`, `std`, `argmax`, `cumsum`, `all`, `any`, axis parameter      |
| 06  | [Random Numbers](06-random-numbers/)                         | `Rng.with_key`, `Rng.uniform`, `Rng.normal`, `Rng.permutation`       |
| 07  | [Linear Algebra](07-linear-algebra/)                         | `*@`, `solve`, `inv`, `det`, `lstsq`, `eigh`, `svd`                  |
| 08  | [Signal Processing](08-signal-processing/)                   | `rfft`, `irfft`, `rfftfreq`, frequency filtering                     |
| 09  | [Image Processing](09-image-processing/)                     | `correlate`, `sliding_window`, Sobel edges, `Nx_io.save_png`         |
| 10  | [Machine Learning](10-machine-learning/)                     | PCA, K-Means, DBSCAN, t-SNE from Nx primitives                       |

## Running

From the repository root:

```bash
dune exec packages/nx/examples/01-creating-arrays/main.exe
dune exec packages/nx/examples/02-infix-and-arithmetic/main.exe
# ... and so on through 10
```

## Dependencies

- Examples 01-08 and 10 use only `nx`
- Example 09 adds `nx.io` (image I/O)
