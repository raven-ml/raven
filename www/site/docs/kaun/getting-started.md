# Getting Started with kaun

This guide shows you how to build and train neural networks with kaun.

## Installation

Kaun isn't released yet. When it is, you'll install it with:

```bash
opam install kaun
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build kaun
```

## Your First Neural Network

Here's a simple example that trains a two-layer network on XOR:

```ocaml
open Kaun

(* Define model architecture *)
module Model = struct
  type t = {
    linear1: Linear.t;
    linear2: Linear.t;
  }

  let create () = 
    let rng = Rng.make 42 in
    {
      linear1 = Linear.create rng ~input_dim:2 ~output_dim:8;
      linear2 = Linear.create rng ~input_dim:8 ~output_dim:1;
    }

  let forward model x =
    x
    |> Linear.forward model.linear1
    |> Activation.relu
    |> Linear.forward model.linear2
    |> Activation.sigmoid
end

(* Training *)
let train () =
  (* XOR dataset *)
  let x = Tensor.of_float_list [|4; 2|] [0.; 0.; 0.; 1.; 1.; 0.; 1.; 1.] in
  let y = Tensor.of_float_list [|4; 1|] [0.; 1.; 1.; 0.] in
  
  (* Initialize model and optimizer *)
  let model = Model.create () in
  let optimizer = Optimizer.adam ~lr:0.01 () in
  
  (* Training loop *)
  let rec train_step model opt_state step =
    if step >= 1000 then model
    else
      (* Forward pass and loss *)
      let loss_fn params =
        let pred = Model.forward params x in
        Loss.sigmoid_binary_cross_entropy ~targets:y pred
      in
      
      (* Compute gradients *)
      let loss, grads = value_and_grad loss_fn model in
      
      (* Update parameters *)
      let model', opt_state' = Optimizer.update optimizer opt_state model grads in
      
      (* Print progress *)
      if step mod 100 = 0 then
        Printf.printf "Step %d, Loss: %.4f\n" step (Tensor.to_float loss);
      
      train_step model' opt_state' (step + 1)
  in
  
  train_step model (Optimizer.init optimizer model) 0
```

## Key Concepts

**Models are records.** Unlike PyTorch's classes, kaun models are OCaml records containing layers. This makes them immutable, parameter updates create new model instances.

**Functional design.** Everything is a function. Models transform inputs to outputs. Optimizers transform (model, gradients) to new models.

**Explicit parameter trees.** Models can be converted to/from parameter trees using lenses. This enables flexible parameter manipulation and will enable serialization.

**Stateful optimizers.** Optimizers like Adam maintain state (momentum, variance). The state is separate from the model and updated alongside it.

## Available Components

```ocaml
(* Layers *)
Linear.create rng ~input_dim:784 ~output_dim:128 ~use_bias:true

(* Activations *)
Activation.relu x
Activation.sigmoid x
Activation.tanh x
Activation.elu ~alpha:1.0 x
Activation.leaky_relu ~negative_slope:0.01 x

(* Initializers *)
Initializer.constant 0.0
Initializer.glorot_uniform

(* Optimizers *)
Optimizer.sgd ~lr:0.01 ()
Optimizer.adam ~lr:0.001 ~beta1:0.9 ~beta2:0.999 ~eps:1e-8 ()

(* Loss functions *)
Loss.sigmoid_binary_cross_entropy ~targets pred
```

## Design Patterns

**Module-based models:**
```ocaml
module MyModel = struct
  type t = { 
    conv1: Conv2d.t;  (* not implemented yet *)
    fc1: Linear.t;
    (* ... *)
  }
  
  let create rng = (* ... *)
  let forward t x = (* ... *)
end
```

**Lens-based parameter access:**
```ocaml
(* Convert model to parameters *)
let params = to_ptree model

(* Update specific parameters *)
let new_params = Ptree.map (fun t -> Tensor.mul_scalar t 0.9) params

(* Convert back to model *)
let new_model = of_ptree new_params
```

## Next Steps

- [MNIST Tutorial](/docs/kaun/mnist-tutorial/) - Train a real CNN on image data
- Check out the examples in `kaun/example/` for more complete training loops

Kaun is under active development. More layers, losses, and utilities are coming.