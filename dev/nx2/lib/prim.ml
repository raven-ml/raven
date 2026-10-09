(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value
module L = Nx_array.Layout

let first : type v s d. (v, s, d) t -> (v, s) Nx_array.t = function
  | Array { a; _ } -> a
  | Shards { arrays; _ } -> arrays.(0)

let placement : type v s d. (v, s, d) t -> d Devices.placement = function
  | Array { at; _ } | Shards { at; _ } -> at

let dtype x = Nx_array.dtype (first x)
let rank x = L.rank (Nx_array.layout (first x))

(* The tiles along [axis] of a value at [p]: 1 where it is not cut. *)
let tiles p axis =
  Array.fold_left
    (fun n (a, t) -> if a = axis then t else n)
    1
    (Grid.cuts (Devices.grid p))

let dim (type v s d) (x : (v, s, d) t) i =
  let l = Nx_array.layout (first x) in
  if i < 0 || i >= L.rank l then
    invalid_arg
      (Printf.sprintf "Prim.dim: axis %d of a value of rank %d" i (L.rank l));
  match x with
  | Array _ -> L.dim l i
  | Shards { at; _ } -> L.dim l i * tiles at i

let shape x = Array.init (rank x) (dim x)

let form x =
  match x with
  | Array { at; a } ->
      { dtype = Nx_array.dtype a; layout = Nx_array.layout a; placement = at }
  | Shards { at; _ } ->
      { dtype = dtype x; layout = L.contiguous (shape x); placement = at }
