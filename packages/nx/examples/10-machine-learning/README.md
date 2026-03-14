# Machine Learning

Four classic ML algorithms built from Nx primitives: SVD, broadcasting,
reductions, and scalar loops.

| File       | Algorithm | Key Nx operations                                         |
| ---------- | --------- | --------------------------------------------------------- |
| `pca.ml`   | PCA       | `svd`, `mean`, `matmul`, `cumsum`                         |
| `kmeans.ml`| K-Means   | broadcasting, `argmin`, `categorical`, `sq_distances`     |
| `dbscan.ml`| DBSCAN    | pairwise distances, `less_equal_s`, boolean `item` in BFS |
| `tsne.ml`  | t-SNE     | `exp`, `log`, Student-t kernel, momentum gradient descent |

## Running

```bash
dune exec nx/examples/10-machine-learning/pca.exe
dune exec nx/examples/10-machine-learning/kmeans.exe
dune exec nx/examples/10-machine-learning/dbscan.exe
dune exec nx/examples/10-machine-learning/tsne.exe
```
