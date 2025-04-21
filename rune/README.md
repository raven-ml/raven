# Rune

JAX-inspired automatic differentiation and JIT compilation library for OCaml

Rune brings JAX-like capabilities to OCaml, enabling high-performance numerical
computation with automatic differentiation, multi-device support (CPU, CUDA,
Metal), and JIT compilation.

## Features

- N-dimensional tensor operations (arithmetic, linear algebra, etc.)
- Automatic differentiation: `grad`, `grads`, `value_and_grad`, `value_and_grads`
- Functional API for pure computations
- Multi-device backends: CPU, CUDA, Metal
- Random tensor initialization: `rand`
- JIT compilation to accelerate operations on GPU backends
- Seamless interop with Ndarray for data loading and visualization

## Quick Start

```ocaml
open Rune

(* Define a simple function: sum of squares *)
let f x = sum (mul x x)

(* Create input tensor *)
let x = create Float32 [|3;3|] (Array.init 9 float_of_int)

(* Compute gradient of f at x *)
let grad_x = grad f x

(* Print gradient *)
print grad_x
```

## Examples

See the `example/` directory for:
- `01-mlp`: training a simple MLP with `value_and_grads`
- `xx-higher-derivative`: computing higher-order derivatives

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
