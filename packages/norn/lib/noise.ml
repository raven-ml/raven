(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let at k = Nx.beside (Nx.Rng.to_tensor k)
let normal k dt shape = Nx.Rng.normal_at (at k) ~key:k dt shape
let uniform k dt shape = Nx.Rng.uniform_at (at k) ~key:k dt shape
let exponential k dt shape = Nx.Rng.exponential_at (at k) ~key:k dt shape
let gumbel k dt shape = Nx.Rng.gumbel_at (at k) ~key:k dt shape

let normal_like (type a b) (x : (a, b, 'd) Nx.t) k shape : (a, b, 'd) Nx.t =
  let draw (type f) (x : (float, f, 'd) Nx.t) = normal k (Nx.dtype x) shape in
  match Nx.Dtype.kind (Nx.dtype x) with
  | Float -> draw x
  | _ -> Nx.zeros_at (at k) (Nx.dtype x) shape
