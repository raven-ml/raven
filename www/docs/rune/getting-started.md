# Getting Started with rune

This guide shows you how to use automatic differentiation and JIT compilation with rune.

## Installation

Rune isn't released yet. When it is, you'll install it with:

```bash
opam install rune
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build rune
```

## Your First Gradient

Here's a simple example computing the gradient of a function:

```ocaml
open Rune

(* Define a function *)
let f x =
  add (mul x x) (sin x)

(* Compute its gradient *)
let f' = grad f

(* Evaluate at a point *)
let () =
  let x = scalar cpu Float32 2.0 in
  let y = f x in
  let dy_dx = f' x in
  
  Printf.printf "f(2) = %.4f\n" (unsafe_get y [||]);
  Printf.printf "f'(2) = %.4f\n" (unsafe_get dy_dx [||])
```

## Key Concepts

**Everything is at the top level.** When you `open Rune`, all tensor operations are available directly, no submodules needed.

**Device contexts.** Tensors are created with a device context:

```ocaml
(* CPU tensors *)
let x = rand cpu Float32 [|100|] ~from:(-1.) ~to_:1.

(* Metal tensors (macOS only) *)
let gpu_device = metal () in
let y = rand gpu_device Float32 [|100|] ~from:(-1.) ~to_:1.
```

**Composable transformations.** Just like JAX, you can compose transformations:

```ocaml
(* Second derivative *)
let f'' = grad (grad f)

(* JIT-compiled gradient *)
let fast_grad = jit (grad f)
```

## Computing Gradients

For machine learning, you typically need gradients of a loss function:

```ocaml
(* Simple linear model *)
let linear_model w b x =
  add (mul w x) b

(* Mean squared error loss *)
let loss_fn params x_batch y_batch =
  let w = slice params [R (0, 1)] in
  let b = slice params [R (1, 2)] in
  let predictions = linear_model w b x_batch in
  let errors = sub predictions y_batch in
  mean (mul errors errors)

(* Get gradient function *)
let grad_fn = grad loss_fn

(* Training step *)
let train_step params x_batch y_batch learning_rate =
  let grads = grad_fn params x_batch y_batch in
  sub params (mul (scalar cpu Float32 learning_rate) grads)
```

## JIT Compilation

JIT compilation makes functions faster by tracing and optimizing them:

```ocaml
(* Original function *)
let matmul_relu a b =
  let c = matmul a b in
  maximum c (zeros_like c)

(* JIT compiled version *)
let fast_matmul_relu = jit matmul_relu

(* First call traces and compiles, subsequent calls use cached kernel *)
let result = fast_matmul_relu a b
```

JIT compilation is shape-specialized. Different input shapes trigger recompilation.

## Neural Network Example

Here's a simple two-layer network:

```ocaml
let mlp w1 b1 w2 b2 x =
  (* First layer *)
  let h = add (matmul x w1) b1 in
  let h = maximum h (zeros_like h) in  (* ReLU *)
  
  (* Second layer *)
  add (matmul h w2) b2

(* Initialize parameters *)
let init_mlp input_dim hidden_dim output_dim =
  let w1 = rand cpu Float32 [|input_dim; hidden_dim|] ~from:(-0.1) ~to_:0.1 in
  let b1 = zeros cpu Float32 [|hidden_dim|] in
  let w2 = rand cpu Float32 [|hidden_dim; output_dim|] ~from:(-0.1) ~to_:0.1 in
  let b2 = zeros cpu Float32 [|output_dim|] in
  (w1, b1, w2, b2)

(* Compute loss and gradients *)
let loss params x y =
  let w1, b1, w2, b2 = params in
  let pred = mlp w1 b1 w2 b2 x in
  mean (mul (sub pred y) (sub pred y))

let train_step params x y lr =
  let grad_loss = grads loss in
  let [gw1; gb1; gw2; gb2] = grad_loss [w1; b1; w2; b2] x y in
  let w1' = sub w1 (mul (scalar cpu Float32 lr) gw1) in
  let b1' = sub b1 (mul (scalar cpu Float32 lr) gb1) in
  let w2' = sub w2 (mul (scalar cpu Float32 lr) gw2) in
  let b2' = sub b2 (mul (scalar cpu Float32 lr) gb2) in
  (w1', b1', w2', b2')
```

## Advanced Features

**Higher-order derivatives:**
```ocaml
let f x = add (mul x (mul x x)) (mul x x) in
let f' = grad f in
let f'' = grad f' in
let f''' = grad f'' in

let x = scalar cpu Float32 2.0 in
Printf.printf "f'''(2) = %.4f\n" (unsafe_get (f''' x) [||])  (* Should be 6.0 *)
```

**Multiple inputs with grads:**
```ocaml
let f [x; y] = add (mul x x) (mul y y) in
let df = grads f in
let [dx; dy] = df [scalar cpu Float32 3.0; scalar cpu Float32 4.0] in
(* dx = 6.0, dy = 8.0 *)
```

## Design Notes

**Why separate Tensor from nx arrays?** Tensors carry gradient information and device placement. Nx arrays are just data. This separation keeps nx simple while enabling autodiff in rune.

**Effect-based autodiff.** Rune uses OCaml 5's effects to implement autodiff without macros or operator overloading. This gives us clean syntax and composable transformations.

## Next Steps

Rune is the foundation for the entire deep learning stack. Once you understand gradients and JIT, you can:

- Use Kaun for high-level neural network abstractions
- Apply Sowilo's differentiable image processing
- Train models interactively in Quill notebooks

Check out the examples in `rune/example/` for complete neural network implementations including MNIST classification.