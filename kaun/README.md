# Kaun

> Status: **Exploration** 🔍

A Flax-inspired neural network library for OCaml, built on Rune's automatic differentiation engine.

## Vision

Kaun brings modern deep learning to OCaml with a flexible, type-safe API for building and training neural networks. It leverages Rune for automatic differentiation and computation graph optimization while maintaining OCaml's functional programming advantages.

## Current Status

Kaun is in the **Exploration** phase with core API design in development and limited functionality that's subject to significant changes.

## Features (Planned)

- Flax-like functional API for neural networks
- Common layers (Dense, Conv2D, BatchNorm)
- Standard optimizers (SGD, Adam)
- Weight initialization strategies
- Training utilities and model management

## Getting Started

**Note:** As Kaun is in the exploration phase, these instructions are preliminary and subject to change.

### Example Usage (Aspirational)

```ocaml
open Kaun

(* Define a simple MLP *)
let mlp = 
  Sequential.make [
    Dense.make ~units:128 ~activation:Activation.relu;
    Dense.make ~units:64 ~activation:Activation.relu;
    Dense.make ~units:10 ~activation:Activation.softmax;
  ]

(* Initialize parameters *)
let rng, params = Random.make_key 42 |> mlp.init_params

(* Create optimizer *)
let optimizer = Optimizer.adam ~lr:0.001

(* Training step *)
let train_step params x y opt_state =
  let loss, grads = Rune.value_and_grad loss_fn params x y in
  let updates, opt_state = Optimizer.step optimizer opt_state params grads in
  let params = Optimizer.apply_updates params updates in
  (params, opt_state, loss)
```

## Relationship to Rune and Raven

Kaun builds on top of Rune to provide a higher-level API for deep learning:

- **Rune** provides the automatic differentiation foundation
- **Kaun** builds higher-level neural network abstractions
