(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Per-sample gradients by composing vmap with grad: write the loss for one
   example, differentiate it, and map the differentiated function over the
   batch. Each parameter leaf gains a leading batch axis. *)

(* Model parameters: closed over by the mapped function, so they are constants
   of the map and gradients are taken with respect to them. *)
type params = { w : Nx.float32_t; b : Nx.float32_t }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
    { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

(* One example: an input row and its target. vmap maps over axis 0 of both
   leaves, so the mapped function sees a single row and a scalar target. *)
type example = { x : Nx.float32_t; y : Nx.float32_t }

module Example = struct
  type t = example

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { x; y } =
    { x = f x; y = f y }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { x = f a.x b.x; y = f a.y b.y }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { x; y } =
    f x;
    f y
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let n, d = (8, 3) in
  let params =
    { w = Nx.randn Nx.float32 [| d |]; b = Nx.randn Nx.float32 [||] }
  in
  let batch =
    { x = Nx.randn Nx.float32 [| n; d |]; y = Nx.randn Nx.float32 [| n |] }
  in

  (* Squared error of a linear model on a single example. *)
  let loss ex p =
    let pred = Nx.add (Nx.dot ex.x p.w) p.b in
    Nx.square (Nx.sub pred ex.y)
  in

  (* grad gives the per-example gradient function; vmap2 maps it over the batch.
     The result has the parameters' structure with a leading batch axis on every
     leaf: w is [n; d] and b is [n]. *)
  let per_sample =
    Rune.vmap2
      (module Example)
      (module Params)
      (fun ex -> Rune.grad (module Params) (loss ex) params)
      batch
  in
  Printf.printf "per-sample dw: %s\n"
    (Nx.shape_to_string (Nx.shape per_sample.w));
  Printf.printf "per-sample db: %s\n\n"
    (Nx.shape_to_string (Nx.shape per_sample.b));

  (* The same thing, one example at a time. *)
  let row i t = Nx.slice [ Nx.I i ] t in
  let looped i =
    Rune.grad
      (module Params)
      (loss { x = row i batch.x; y = row i batch.y })
      params
  in
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
  Printf.printf "per-sample |dw|: %s\n" (Nx.data_to_string norms)
