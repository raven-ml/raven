(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let normal_like (type a b) (x : (a, b) Nx.t) k shape : (a, b) Nx.t =
  let draw (type f) (x : (float, f) Nx.t) =
    Nx.Rng.normal k (Nx.dtype x) shape
  in
  match Nx_dtype.kind (Nx.dtype x) with
  | Float -> draw x
  | _ -> Nx.zeros (Nx.dtype x) shape
