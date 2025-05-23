# Getting Started with Raven

Welcome to Raven, the comprehensive OCaml ecosystem for machine learning and data science. This guide will help you get up and running quickly.

## Installation

### Using OPAM

The easiest way to install Raven is through OPAM:

```bash
# Install core libraries
opam install ndarray hugin

# For GPU acceleration (optional)
opam install ndarray-metal  # macOS
opam install ndarray-cuda   # Linux
```

### From Source

To build from source for the latest features:

```bash
git clone https://github.com/your-org/raven.git
cd raven
dune build
```

## Your First Array

Let's start with creating and manipulating arrays using ndarray:

```ocaml
open Ndarray

(* Create a 3x3 matrix of ones *)
let a = ones [3; 3] float32

(* Create a random matrix *)
let b = random [3; 3] float32

(* Matrix multiplication *)
let c = matmul a b

(* Element-wise operations *)
let d = add c (scalar 2.0 float32)

(* Reductions *)
let mean_val = mean d
let sum_cols = sum d ~axis:[0]
```

## Creating Your First Plot

Use Hugin to visualize your data:

```ocaml
open Hugin

(* Generate some data *)
let x = Ndarray.linspace 0.0 10.0 100 float64
let y = Ndarray.sin x

(* Create a plot *)
let fig = Figure.create ()
let ax = Figure.add_subplot fig 1 1 1
let _ = Axes.plot ax x y ~label:"sin(x)"

(* Customize *)
let _ = Axes.set_title ax "Sine Wave"
let _ = Axes.set_xlabel ax "x"
let _ = Axes.set_ylabel ax "y"
let _ = Axes.legend ax ()

(* Save the plot *)
let _ = Figure.savefig fig "sine_wave.png"
```

## Next Steps

Now that you have the basics, explore more advanced features:

- **[Array Operations](array-operations.html)** - Learn about broadcasting, indexing, and advanced operations
- **[Linear Algebra](linear-algebra.html)** - Matrix operations, decompositions, and solvers
- **[Plotting Guide](plotting.html)** - Create beautiful visualizations with Hugin
- **[Auto-differentiation](autodiff.html)** - Get started with Rune for machine learning

## Getting Help

- Check out the [API documentation](/api/)
- Browse [examples](/examples/)
- Join the [community discussions](https://github.com/your-org/raven/discussions)
- Report [issues](https://github.com/your-org/raven/issues)