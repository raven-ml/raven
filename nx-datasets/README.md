# nx-datasets

Dataset loading utilities for Nx

`nx-datasets` offers easy access to popular machine learning and data
science datasets as Nx tensors, with automatic download and caching.

## Features

- Automatic download and local caching of datasets
- Returns data as Nx tensors ready for training and evaluation
- Supported datasets:
  - MNIST (handwritten digits, 28×28 grayscale)
  - Fashion‑MNIST (Zalando article images)
  - CIFAR‑10 (32×32 color images)
  - Iris flower dataset (150 samples, 4 features)
  - Breast Cancer Wisconsin dataset
  - Diabetes regression dataset
  - California Housing regression dataset
  - Airline Passengers time series

## Quick Start

```ocaml
open Nx_datasets

(* Load MNIST dataset *)
let (x_train, y_train), (x_test, y_test) = load_mnist ()

(* x_train : uint8 [|60000; 28; 28; 1|], y_train : uint8 [|60000; 1|] *)
(* x_test  : uint8 [|10000; 28; 28; 1|], y_test  : uint8 [|10000; 1|] *)
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
