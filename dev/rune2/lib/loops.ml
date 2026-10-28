(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module D = Nx.Dtype

let contiguous shape = Nx_array.Layout.contiguous (Iarray.of_array shape)

let shape_of (f : _ Nx.Prim.form) =
  Iarray.to_array (Nx_array.Layout.shape f.layout)

let row (V x) =
  let s = Nx.shape x in
  F
    {
      dtype = Nx.dtype x;
      layout = contiguous (Array.sub s 1 (Array.length s - 1));
      placement = Nx.Placement.without_leading_axis (Nx.placement x);
    }

(* Values *)

let neg_zero (type v s) (dt : (v, s) D.t) : v =
  match D.kind dt with
  | Float -> -0.
  | Complex -> { Complex.re = -0.; im = -0. }
  | Signed | Unsigned | Boolean -> D.zero dt

let filled ~zero (F f) =
  let v = if zero then D.zero f.dtype else neg_zero f.dtype in
  V (Nx.full_at f.placement f.dtype (shape_of f) v)

let zeros_like (V x) = V (Nx.zeros_like x)
let flip (V x) = V (Nx.flip ~axes:[ 0 ] x)

let add (V a) b = V (Nx.add a (like a b))

let rec holds_iterate (p : Program.t) =
  Array.exists
    (function
      | Op _ -> false
      | Do { c = Iterate _; _ } -> true
      | Do { c; _ } -> List.exists holds_iterate (carried c))
    p.eqns

(* Programs *)

let reorder (p : Program.t) order =
  Program.make ~owner:p.owner
    ~params:(Array.map (fun k -> p.params.(k)) order)
    (Array.to_list p.eqns) ~outputs:p.outputs ~adds:p.adds ~slots:p.slots

let pick mask a =
  Array.of_list (List.filteri (fun k _ -> mask.(k)) (Array.to_list a))

let split lengths a =
  let rec go at = function
    | [] ->
        if at <> Array.length a then invalid_arg "Loops.split";
        []
    | n :: rest -> Array.sub a at n :: go (at + n) rest
  in
  go 0 lengths

(* [vs] where [mask] holds, and [ws] elsewhere, each in order. *)
let merge mask vs ws =
  let a = ref (-1) and b = ref (-1) in
  Array.map
    (fun m ->
      if m then begin
        incr a;
        vs.(!a)
      end
      else begin
        incr b;
        ws.(!b)
      end)
    mask
