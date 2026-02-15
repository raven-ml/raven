# Nx_datasets

Load common machine learning datasets and generate synthetic data for testing.

## Overview

Nx_datasets provides two categories of data:
- **Real datasets**: Downloaded and cached locally (MNIST, CIFAR-10, Iris, etc.)
- **Synthetic generators**: Create data on-the-fly for testing (blobs, moons, regression problems)

Real datasets are automatically downloaded on first use and cached in your platform's cache directory.


## Available Datasets Reference

### Real Datasets

| Dataset            | Function                  | Samples | Features | Task           |
| ------------------ | ------------------------- | ------- | -------- | -------------- |
| [MNIST](#mnist)              | `load_mnist`              | 70,000  | 28×28×1  | Classification |
| [Fashion-MNIST](#fashion-mnist)      | `load_fashion_mnist`      | 70,000  | 28×28×1  | Classification |
| [CIFAR-10](#cifar-10)           | `load_cifar10`            | 60,000  | 32×32×3  | Classification |
| [Iris](#iris)               | `load_iris`               | 150     | 4        | Classification |
| [Breast Cancer](#breast-cancer)      | `load_breast_cancer`      | 569     | 30       | Classification |
| [Diabetes](#regression-datasets)           | `load_diabetes`           | 442     | 10       | Regression     |
| [California Housing](#regression-datasets) | `load_california_housing` | 20,640  | 8        | Regression     |
| [Airline Passengers](#time-series) | `load_airline_passengers` | 144     | 1        | Time Series    |

### Synthetic Generators

| Generator | Function | Purpose | Parameters |
| --------- | -------- | ------- | ---------- |
| [Gaussian Blobs](#gaussian-blobs) | `make_blobs` | Clustering | centers, cluster_std |
| [Two Moons](#two-moons) | `make_moons` | Non-linear classification | noise, n_samples |
| [Concentric Circles](#concentric-circles) | `make_circles` | Non-linear classification | noise, factor |
| [Classification](#complex-classification) | `make_classification` | Controlled features | n_informative, n_redundant |
| [Regression](#linear-regression) | `make_regression` | Linear relationships | noise, n_features |
| [Friedman](#friedman-benchmarks) | `make_friedman1/2/3` | Non-linear regression | - |
| [Swiss Roll](#swiss-roll) | `make_swiss_roll` | Manifold learning | n_samples |
| [S-Curve](#s-curve) | `make_s_curve` | Manifold learning | n_samples |

## Loading Real Datasets

### Image Datasets

#### MNIST

Classic handwritten digits dataset:

```ocaml
let (x_train, _y_train), (x_test, _y_test) = Nx_datasets.load_mnist ()
let () =
  Printf.printf "Train: %s, Test: %s\n"
    (Nx.shape_to_string (Nx.shape x_train))
    (Nx.shape_to_string (Nx.shape x_test))
(* Train: [60000, 28, 28, 1], Test: [10000, 28, 28, 1] *)
```

Images are uint8 arrays with values 0-255. Labels are single digits 0-9.

#### Fashion-MNIST

Clothing classification with the same format as MNIST:

```ocaml
let (_x_train, _y_train), (_x_test, _y_test) = Nx_datasets.load_fashion_mnist ()
(* 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot *)
```

#### CIFAR-10

Color images in 10 categories:

```ocaml
let (_x_train, _y_train), (_x_test, _y_test) = Nx_datasets.load_cifar10 ()
(* x_train shape: [50000, 32, 32, 3] *)
(* Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck *)
```

### Tabular Datasets

#### Iris

Classic flower classification:

```ocaml
let _x, _y = Nx_datasets.load_iris ()
(* x shape: [150, 4] - sepal length/width, petal length/width *)
(* y shape: [150, 1] - 0=setosa, 1=versicolor, 2=virginica *)
```

#### Breast Cancer

Binary classification for cancer diagnosis:

```ocaml
let _x, _y = Nx_datasets.load_breast_cancer ()
(* x shape: [569, 30] - 30 features per sample *)
(* y shape: [569, 1] - 0=malignant, 1=benign *)
```

#### Regression Datasets

```ocaml
(* Diabetes regression *)
let _x, _y = Nx_datasets.load_diabetes ()
(* x: [442, 10], y: [442, 1] - diabetes progression *)

(* California housing prices *)
let _x, _y = Nx_datasets.load_california_housing ()
(* x: [20640, 8], y: [20640, 1] - median house values *)
```

### Time Series

```ocaml
let _passengers = Nx_datasets.load_airline_passengers ()
(* Monthly airline passenger counts 1949-1960 *)
(* shape: [144] *)
```

## Generating Synthetic Data

### Classification Datasets

#### Gaussian Blobs

Generate isotropic Gaussian blobs for clustering:

```ocaml
let _x, _y = Nx_datasets.make_blobs
  ~n_samples:300
  ~centers:(`N 3)
  ~cluster_std:0.5
  ()
(* 3 well-separated clusters *)
```

Specify exact cluster centers:

```ocaml
let centers = Nx.create Nx.float32 [|3; 2|]
  [|-10.; -10.; 0.; 0.; 10.; 10.|]
let _x, _y = Nx_datasets.make_blobs ~centers:(`Array centers) ()
```

#### Two Moons

Binary classification with interleaving half circles:

```ocaml
let _x, _y = Nx_datasets.make_moons
  ~n_samples:200
  ~noise:0.1
  ()
(* Ideal for testing non-linear classifiers *)
```

#### Concentric Circles

Nested circles for non-linear separation:

```ocaml
let _x, _y = Nx_datasets.make_circles
  ~n_samples:200
  ~noise:0.05
  ~factor:0.5  (* Inner circle radius ratio *)
  ()
```

#### Complex Classification

Control informative/redundant features:

```ocaml
let _x, _y = Nx_datasets.make_classification
  ~n_samples:1000
  ~n_features:20
  ~n_informative:15  (* Useful features *)
  ~n_redundant:5     (* Linear combinations *)
  ~n_classes:3
  ~n_clusters_per_class:2
  ()
```

### Regression Datasets

#### Linear Regression

Generate data with controllable properties:

```ocaml
let _x, _y, _coef_opt = Nx_datasets.make_regression
  ~n_samples:100
  ~n_features:5
  ~n_informative:3  (* Only 3 features affect output *)
  ~noise:10.0       (* Gaussian noise std dev *)
  ~coef:true        (* Return true coefficients *)
  ()
```

#### Friedman Benchmarks

Standard non-linear regression problems:

```ocaml
(* Friedman #1: y = 10*sin(π*x1*x2) + 20*(x3-0.5)² + 10*x4 + 5*x5 + noise *)
let _x, _y = Nx_datasets.make_friedman1 ~n_samples:100 ()
```

### Manifold Data

#### Swiss Roll

3D manifold for dimensionality reduction:

```ocaml
let _x, _color = Nx_datasets.make_swiss_roll ~n_samples:1000 ()
(* x shape: [1000, 3], color: [1000] - position along roll *)
```

#### S-Curve

Another 3D manifold:

```ocaml
let _x, _color = Nx_datasets.make_s_curve ~n_samples:1000 ()
```
