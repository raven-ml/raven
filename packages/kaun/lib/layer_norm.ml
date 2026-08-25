(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { gamma : 'a; beta : 'a }

let map f { gamma; beta } =
  let gamma = f gamma in
  let beta = f beta in
  { gamma; beta }

let map2 f p q =
  let gamma = f p.gamma q.gamma in
  let beta = f p.beta q.beta in
  { gamma; beta }

let iter f { gamma; beta } =
  f gamma;
  f beta

let fold f acc { gamma; beta } = f "beta" (f "gamma" acc gamma) beta
let fold2 f acc p q = f "beta" (f "gamma" acc p.gamma q.gamma) p.beta q.beta
let names _ = { gamma = "gamma"; beta = "beta" }

let make ~dim dtype =
  if dim <= 0 then
    Printf.ksprintf invalid_arg "Layer_norm.make: dim must be positive, got %d"
      dim;
  { gamma = Nx.ones dtype [| dim |]; beta = Nx.zeros dtype [| dim |] }

let init ~dim = make ~dim Nx.float32

(* Half and quarter precision floats are too coarse for the statistics: their
   normalization runs in a float32 island. Wider dtypes keep their own
   arithmetic, so the float32 and float64 graphs are exactly the pre-island
   ones. *)
let low_precision : type b. (float, b) Nx.dtype -> bool = function
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> true
  | Nx.Float32 | Nx.Float64 -> false

(* [(x - mean x) / sqrt (var x + eps)] along the last axis. Biased variance of
   the centered values: mean((x - mu)^2). Centering before squaring keeps the
   computation stable for large offsets. *)
let normalize ~eps x =
  let axes = [ Array.length (Nx.shape x) - 1 ] in
  let mu = Nx.mean ~axes ~keepdims:true x in
  let xc = Nx.sub x mu in
  let var = Nx.mean ~axes ~keepdims:true (Nx.mul xc xc) in
  Nx.div xc (Nx.sqrt (Nx.add_s var eps))

let apply ?(eps = 1e-5) { gamma; beta } x =
  if eps < 0.0 then
    Printf.ksprintf invalid_arg "Layer_norm.apply: eps must be >= 0, got %g" eps;
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank = 0 then invalid_arg "Layer_norm.apply: input must not be a scalar";
  let dim = (Nx.shape gamma).(0) in
  if shape.(rank - 1) <> dim then
    Printf.ksprintf invalid_arg
      "Layer_norm.apply: last axis has size %d but the layer normalizes %d \
       features"
      shape.(rank - 1)
      dim;
  let dt = Nx.dtype x in
  let normalized =
    if low_precision dt then Nx.cast dt (normalize ~eps (Nx.cast Nx.float32 x))
    else normalize ~eps x
  in
  Nx.add (Nx.mul normalized gamma) beta
