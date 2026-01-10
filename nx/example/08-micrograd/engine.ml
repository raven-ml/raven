(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Engine.ml - Automatic differentiation engine using nx tensors *)

open Nx

type 'a value = {
  mutable data : 'a; (* The actual tensor data *)
  grad : 'a ref; (* Gradient tensor (mutable) *)
  children : 'a value list; (* Dependencies for backprop *)
  op : string; (* Operation that created this value *)
  backward_fn : (unit -> unit) ref; (* Backward pass function *)
}
(** Value type that stores a tensor and its gradient for automatic
    differentiation *)

(** Create a new value from tensor data *)
let create ?(op = "") ?(children = []) data =
  let zero_grad = zeros_like data in
  { data; grad = ref zero_grad; children; op; backward_fn = ref (fun () -> ()) }

(** Create a value from a scalar *)
let scalar dtype value = create (Nx.scalar dtype value)

(** Create a scalar with the same dtype as a tensor *)
let scalar_like tensor value = Nx.full (dtype tensor) (shape tensor) value

(** Addition with automatic differentiation *)
let ( + ) v1 v2 =
  let out_data = add v1.data v2.data in
  let out = create ~op:"+" ~children:[ v1; v2 ] out_data in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       v1.grad := add !(v1.grad) grad_out;
       v2.grad := add !(v2.grad) grad_out);
  out

(** Multiplication with automatic differentiation *)
let ( * ) v1 v2 =
  let out_data = mul v1.data v2.data in
  let out = create ~op:"*" ~children:[ v1; v2 ] out_data in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       v1.grad := add !(v1.grad) (mul v2.data grad_out);
       v2.grad := add !(v2.grad) (mul v1.data grad_out));
  out

(** Power operation *)
let ( ** ) v power =
  let out_data = pow_s v.data power in
  let out =
    create ~op:(Printf.sprintf "**%.2f" power) ~children:[ v ] out_data
  in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       let power_scalar = scalar_like v.data power in
       let derivative =
         mul (mul power_scalar (pow_s v.data (power -. 1.0))) grad_out
       in
       v.grad := add !(v.grad) derivative);
  out

(** ReLU activation with automatic differentiation *)
let relu v =
  let out_data = Nx.relu v.data in
  let out = create ~op:"ReLU" ~children:[ v ] out_data in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       let mask = greater v.data (zeros_like v.data) in
       let mask_float = cast (dtype v.data) mask in
       let derivative = mul mask_float grad_out in
       v.grad := add !(v.grad) derivative);
  out

(** Negation *)
let neg v =
  let neg_one = scalar_like v.data (-1.0) in
  create neg_one * v

(** Reverse addition (for scalar + value) *)
let radd v scalar_val = scalar_val + v

(** Subtraction *)
let ( - ) v1 v2 = v1 + neg v2

(** Reverse subtraction (for scalar - value) *)
let rsub v scalar_val = scalar_val + neg v

(** Reverse multiplication (for scalar * value) *)
let rmul v scalar_val = scalar_val * v

(** Division *)
let ( / ) v1 v2 = v1 * (v2 ** -1.0)

(** Reverse division (for scalar / value) *)
let rdiv v scalar_val = scalar_val * (v ** -1.0)

(** Backward pass implementation *)
let backward v =
  (* Use a hashtable for efficient visited tracking *)
  let module NodeSet = struct
    module H = Hashtbl.Make (struct
      type t = float32_t value

      let equal a b = a == b
      let hash v = Hashtbl.hash (Obj.magic v : int)
    end)

    let create () = H.create 1024
    let mem = H.mem
    let add h v = H.add h v ()
  end in
  (* Topological sort with efficient visited set *)
  let rec topo_sort visited topo v =
    if not (NodeSet.mem visited v) then (
      NodeSet.add visited v;
      let final_topo =
        List.fold_left
          (fun top child -> topo_sort visited top child)
          topo v.children
      in
      v :: final_topo)
    else topo
  in

  let visited = NodeSet.create () in
  let topo_order = topo_sort visited [] v in

  (* Set gradient of output to 1 *)
  v.grad := ones_like v.data;

  (* Apply backward functions in reverse topological order *)
  List.iter (fun node -> !(node.backward_fn) ()) topo_order

(** Zero all gradients *)
let zero_grad values = List.iter (fun v -> v.grad := zeros_like v.data) values

(** Helper function to get scalar value from tensor *)
let item v = Nx.item [] v.data

(** Helper function to get scalar gradient from tensor *)
let grad_item v = Nx.item [] !(v.grad)

(** Pretty print a value *)
let pp_value v =
  let data_val = item v in
  let grad_val = grad_item v in
  Printf.sprintf "Value(data=%.4f, grad=%.4f)" data_val grad_val

(** Print a value *)
let print_value v = Printf.printf "%s\n" (pp_value v)

(** Convert value to string representation *)
let to_string v = pp_value v
