(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Per-sample gradients by composing vmap with grad: write the loss for one
   example, differentiate it, and map the differentiated function over the
   batch. Each tensor of the gradient gains a leading batch axis. *)

(* Model parameters: captured by the mapped function, so they are constants of
   the map and gradients are taken with respect to them. *)
type 'a params = { w : 'a; b : 'a }

let shape_to_string s =
  Printf.sprintf "[%s]"
    (String.concat "x" (List.map string_of_int (Array.to_list s)))

module Params = struct
  type 'a t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params_ptree = Nx.Ptree.instantiate (module Params)

let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  let n, d = (8, 3) in
  let params =
    { w = Nx.randn Nx.float32 [| d |]; b = Nx.randn Nx.float32 [||] }
  in
  let xs = Nx.randn Nx.float32 [| n; d |]
  and ys = Nx.randn Nx.float32 [| n |] in

  (* Squared error of a linear model on a single example. *)
  let loss x y p =
    let pred = Nx.add (Nx.dot x p.w) p.b in
    Nx.square (Nx.sub pred y)
  in

  (* grad gives the per-example gradient function; vmap maps it over axis 0 of
     both arguments, one example per lane. The signature says that the mapped
     function takes two tensors and returns the parameters' structure, and every
     tensor of the result gains a leading batch axis: w is [n; d] and b is
     [n]. *)
  let per_sample =
    Rune.vmap
      Nx.Ptree.(tensor @-> tensor @-> returns params_ptree)
      (fun x y -> Rune.grad params_ptree (loss x y) params)
      xs ys
  in
  Printf.printf "per-sample dw: %s\n" (shape_to_string (Nx.shape per_sample.w));
  Printf.printf "per-sample db: %s\n\n"
    (shape_to_string (Nx.shape per_sample.b));

  (* The same thing, one example at a time. *)
  let row i t = Nx.slice [ Nx.I i ] t in
  let looped i = Rune.grad params_ptree (loss (row i xs) (row i ys)) params in
  let max_diff = ref 0.0 in
  for i = 0 to n - 1 do
    let g = looped i in
    let dw = Nx.max (Nx.abs (Nx.sub (row i per_sample.w) g.w)) in
    let db = Nx.abs (Nx.sub (Nx.slice [ Nx.I i ] per_sample.b) g.b) in
    max_diff := max !max_diff (max (Nx.item [] dw) (Nx.item [] db))
  done;
  Printf.printf "max |vmap - loop| over %d examples: %g\n\n" n !max_diff;

  (* Per-sample gradient norms, e.g. for gradient clipping in DP-SGD. *)
  let norms = Nx.sqrt (Nx.sum ~axes:[ 1 ] (Nx.square per_sample.w)) in
  Printf.printf "per-sample |dw|: %s\n" (Nx.to_string norms)
