(** Nn.ml - Neural network components using Engine *)

open Nx

(** Base module type for neural network components *)
module type MODULE = sig
  type t

  val zero_grad : t -> unit
  val parameters : t -> float32_t Engine.value list
end

(** Single neuron implementation *)
module Neuron = struct
  type t = {
    weights : float32_t Engine.value list;
    bias : float32_t Engine.value;
    nonlin : bool;
  }

  (** Create a neuron with nin inputs *)
  let create nin ?(nonlin = true) () =
    Random.self_init ();
    let weights =
      List.init nin (fun _ ->
          let w_val = Random.float 2.0 -. 1.0 in
          (* Random between -1 and 1 *)
          Engine.scalar float32 w_val)
    in
    let bias = Engine.scalar float32 0.0 in
    { weights; bias; nonlin }

  (** Forward pass through neuron *)
  let call neuron inputs =
    if List.length inputs <> List.length neuron.weights then
      failwith "Input size doesn't match neuron weights";

    (* Compute weighted sum: w1*x1 + w2*x2 + ... + bias *)
    let act =
      List.fold_left2
        (fun acc w x -> Engine.(acc + (w * x)))
        neuron.bias neuron.weights inputs
    in

    if neuron.nonlin then Engine.relu act else act

  (** Get all parameters *)
  let parameters neuron = neuron.bias :: neuron.weights

  (** Zero gradients *)
  let zero_grad neuron = Engine.zero_grad (parameters neuron)

  (** String representation *)
  let to_string neuron =
    let activation = if neuron.nonlin then "ReLU" else "Linear" in
    Printf.sprintf "%sNeuron(%d)" activation (List.length neuron.weights)
end

(** Layer of neurons *)
module Layer = struct
  type t = { neurons : Neuron.t list }

  (** Create a layer with nin inputs and nout outputs *)
  let create nin nout ?(nonlin = true) () =
    let neurons = List.init nout (fun _ -> Neuron.create nin ~nonlin ()) in
    { neurons }

  (** Forward pass through layer *)
  let call layer inputs =
    List.map (fun neuron -> Neuron.call neuron inputs) layer.neurons

  (** Get all parameters *)
  let parameters layer = List.flatten (List.map Neuron.parameters layer.neurons)

  (** Zero gradients *)
  let zero_grad layer = List.iter Neuron.zero_grad layer.neurons

  (** String representation *)
  let to_string layer =
    let neuron_strs = List.map Neuron.to_string layer.neurons in
    Printf.sprintf "Layer of [%s]" (String.concat ", " neuron_strs)
end

(** Multi-Layer Perceptron *)
module MLP = struct
  type t = { layers : Layer.t list }

  (** Create MLP with nin inputs and nouts list of layer sizes *)
  let create nin nouts =
    let sizes = nin :: nouts in
    let num_layers = List.length nouts in
    let layers =
      List.mapi
        (fun i nout ->
          let nin = List.nth sizes i in
          let is_output = Int.equal i (num_layers - 1) in
          Layer.create nin nout ~nonlin:(not is_output)
            () (* No activation on output layer *))
        nouts
    in
    { layers }

  (** Forward pass through MLP *)
  let call mlp inputs =
    (* Ensure inputs is a list of values *)
    let input_list =
      match inputs with [] -> failwith "Empty input list" | _ -> inputs
    in

    List.fold_left (fun x layer -> Layer.call layer x) input_list mlp.layers

  (** Get all parameters *)
  let parameters mlp = List.flatten (List.map Layer.parameters mlp.layers)

  (** Zero gradients *)
  let zero_grad mlp = List.iter Layer.zero_grad mlp.layers

  (** String representation *)
  let to_string mlp =
    let layer_strs = List.map Layer.to_string mlp.layers in
    Printf.sprintf "MLP of [%s]" (String.concat ", " layer_strs)
end
