(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b params = { gamma : (float, 'b) Nx.t; beta : (float, 'b) Nx.t }
type t = Nx.float32_elt params

let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { gamma; beta } =
  { gamma = f gamma; beta = f beta }

let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p q =
  { gamma = f p.gamma q.gamma; beta = f p.beta q.beta }

let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { gamma; beta } =
  f gamma;
  f beta

let names _ = [ "gamma"; "beta" ]

let make ~dim dtype =
  if dim <= 0 then
    Printf.ksprintf invalid_arg "Layer_norm.make: dim must be positive, got %d"
      dim;
  { gamma = Nx.ones dtype [| dim |]; beta = Nx.zeros dtype [| dim |] }

let init ~dim = make ~dim Nx.float32

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
  let axes = [ rank - 1 ] in
  let mu = Nx.mean ~axes ~keepdims:true x in
  let xc = Nx.sub x mu in
  (* Biased variance of the centered values: mean((x - mu)^2). Centering before
     squaring keeps the computation stable for large offsets. *)
  let var = Nx.mean ~axes ~keepdims:true (Nx.mul xc xc) in
  let normalized = Nx.div xc (Nx.sqrt (Nx.add_s var eps)) in
  Nx.add (Nx.mul normalized gamma) beta
