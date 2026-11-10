(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, _) t = { gamma : 'a }

let walk c { gamma } = { gamma = Nx.Ptree.Walk.(field c "gamma" leaf gamma) }

let make ~dim dtype at =
  if dim <= 0 then
    Printf.ksprintf invalid_arg "Rms_norm.make: dim must be positive, got %d"
      dim;
  { gamma = Nx.ones_at at dtype [| dim |] }

let init ~dim at = make ~dim Nx.float32 at

let apply ?(eps = 1e-6) { gamma } x =
  if not (eps >= 0.0) then
    Format.kasprintf invalid_arg "Rms_norm.apply: eps must be >= 0, got %a"
      (Nx.Dtype.pp_value Nx.float64)
      eps;
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
  Nx.rms_norm ~eps ~scale:gamma x
