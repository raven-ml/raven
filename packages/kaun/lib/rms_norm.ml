(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a t = { gamma : 'a }

let walk c { gamma } = { gamma = Nx.Ptree.Walk.(field c "gamma" leaf gamma) }

let make ~dim dtype =
  if dim <= 0 then
    Printf.ksprintf invalid_arg "Rms_norm.make: dim must be positive, got %d"
      dim;
  { gamma = Nx.ones dtype [| dim |] }

let init ~dim = make ~dim Nx.float32

(* Half and quarter precision floats are too coarse for the mean square: the
   normalization runs in a float32 island, as in [Layer_norm]. *)
let low_precision : type b. (float, b) Nx.dtype -> bool = function
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 -> true
  | Nx.Float32 | Nx.Float64 -> false

let normalize ~eps x =
  let axes = [ Array.length (Nx.shape x) - 1 ] in
  let ms = Nx.mean ~axes ~keepdims:true (Nx.mul x x) in
  Nx.div x (Nx.sqrt (Nx.add_s ms eps))

let apply ?(eps = 1e-6) { gamma } x =
  if eps < 0.0 then
    Printf.ksprintf invalid_arg "Rms_norm.apply: eps must be >= 0, got %g" eps;
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank = 0 then invalid_arg "Rms_norm.apply: input must not be a scalar";
  let dim = (Nx.shape gamma).(0) in
  if shape.(rank - 1) <> dim then
    Printf.ksprintf invalid_arg
      "Rms_norm.apply: last axis has size %d but the layer normalizes %d \
       features"
      shape.(rank - 1)
      dim;
  let dt = Nx.dtype x in
  let normalized =
    if low_precision dt then Nx.cast dt (normalize ~eps (Nx.cast Nx.float32 x))
    else normalize ~eps x
  in
  Nx.mul normalized gamma
