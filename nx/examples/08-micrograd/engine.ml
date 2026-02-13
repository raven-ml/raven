(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx

type 'a value = {
  mutable data : 'a;
  grad : 'a ref;
  children : 'a value list;
  op : string;
  backward_fn : (unit -> unit) ref;
}

let create ?(op = "") ?(children = []) data =
  let zero_grad = zeros_like data in
  { data; grad = ref zero_grad; children; op; backward_fn = ref (fun () -> ()) }

let scalar dtype value = create (Nx.scalar dtype value)
let scalar_like tensor value = Nx.full (dtype tensor) (shape tensor) value

let ( + ) v1 v2 =
  let out_data = add v1.data v2.data in
  let out = create ~op:"+" ~children:[ v1; v2 ] out_data in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       v1.grad := add !(v1.grad) grad_out;
       v2.grad := add !(v2.grad) grad_out);
  out

let ( * ) v1 v2 =
  let out_data = mul v1.data v2.data in
  let out = create ~op:"*" ~children:[ v1; v2 ] out_data in

  (out.backward_fn :=
     fun () ->
       let grad_out = !(out.grad) in
       v1.grad := add !(v1.grad) (mul v2.data grad_out);
       v2.grad := add !(v2.grad) (mul v1.data grad_out));
  out

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

let neg v =
  let neg_one = scalar_like v.data (-1.0) in
  create neg_one * v

let radd v scalar_val = scalar_val + v
let ( - ) v1 v2 = v1 + neg v2
let rsub v scalar_val = scalar_val + neg v
let rmul v scalar_val = scalar_val * v
let ( / ) v1 v2 = v1 * (v2 ** -1.0)
let rdiv v scalar_val = scalar_val * (v ** -1.0)

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

let zero_grad values = List.iter (fun v -> v.grad := zeros_like v.data) values
let item v = Nx.item [] v.data
let grad_item v = Nx.item [] !(v.grad)

let pp_value v =
  let data_val = item v in
  let grad_val = grad_item v in
  Printf.sprintf "Value(data=%.4f, grad=%.4f)" data_val grad_val

let print_value v = Printf.printf "%s\n" (pp_value v)
let to_string v = pp_value v
