# `10-data-pipeline`

A complete data preparation pipeline — load, clean, transform, and split. This
example loads the Iris dataset, inspects its statistics, standardizes features,
splits into train/test sets, and runs a nearest-centroid classifier to verify
the pipeline produces usable data.

```bash
dune exec nx/examples/10-data-pipeline/main.exe
```

## What You'll Learn

- Loading datasets with `Nx_datasets.load_iris`
- Inspecting shape, dtype, and per-feature statistics
- Standardizing features with z-score normalization
- Building train/test splits with shuffled indices
- Boolean masking with `compress` and `equal_s`
- Implementing a nearest-centroid classifier from scratch

## Key Functions

| Function                       | Purpose                                          |
| ------------------------------ | ------------------------------------------------ |
| `Nx_datasets.load_iris`        | Load the Iris dataset (150 samples, 4 features)  |
| `squeeze t`                    | Remove singleton dimensions                      |
| `standardize ~axes t`          | Z-score normalization (zero mean, unit variance) |
| `mean ~axes t` / `std ~axes t` | Per-axis mean and standard deviation             |
| `min ~axes t` / `max ~axes t`  | Per-axis minimum and maximum                     |
| `take ~axis indices t`         | Select rows by index array                       |
| `equal_s t scalar`             | Element-wise equality with scalar (returns bool) |
| `compress ~axis ~condition t`  | Keep rows where condition is true                |
| `argmin t`                     | Index of the minimum value                       |
| `set_item indices value t`     | Update a single element                          |

## Output Walkthrough

### Dataset inspection

```
Iris dataset loaded
  Features: [150; 4]  (float64)
  Labels:   [150]     (int32)

Feature statistics:
  sepal length   mean=5.84  std=0.83  min=4.30  max=7.90
  sepal width    mean=3.06  std=0.44  min=2.00  max=4.40
  petal length   mean=3.76  std=1.77  min=1.00  max=6.90
  petal width    mean=1.20  std=0.76  min=0.10  max=2.50
```

### Standardization

After `standardize ~axes:[0]`, each feature has mean near 0 and std near 1:

```
After standardization:
  Column means ~ 0: [-0.00, -0.00, -0.00, 0.00]
  Column stds  ~ 1: [1.00, 1.00, 1.00, 1.00]
```

### Nearest-centroid classifier

For each class, compute the mean of all training samples (the centroid).
To predict, assign each test point to the class of the nearest centroid:

```ocaml
let mask = equal_s y_train (Int32.of_int c) in
let class_samples = compress ~axis:0 ~condition:mask x_train in
let centroid = mean ~axes:[ 0 ] class_samples
```

```
Nearest-centroid accuracy: 80.0% (24/30)
```

## Try It

1. Replace `standardize` with min-max normalization:
   `(x - min ~axes:[0] x) / (max ~axes:[0] x - min ~axes:[0] x)`
2. Print per-class sample counts in the training set to check the split
   is balanced.
3. Try a different random seed for the shuffle and observe how accuracy varies.

## Where to Go From Here

This example ties together everything from the previous nine examples into a
practical ML pipeline. For deep learning with automatic differentiation, explore
the **rune** library — Raven's JAX-equivalent built on Nx.
