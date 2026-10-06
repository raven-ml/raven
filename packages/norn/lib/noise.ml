(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let normal_like (type a b) (x : (a, b) Nx.t) k shape : (a, b) Nx.t =
  let draw (type f) (x : (float, f) Nx.t) =
    Nx.Rng.normal k (Nx.dtype x) shape
  in
  match Nx.dtype x with
  | Nx_dtype.Float64 -> draw x
  | Nx_dtype.Float32 -> draw x
  | Nx_dtype.Float16 -> draw x
  | Nx_dtype.BFloat16 -> draw x
  | Nx_dtype.Float8_e4m3 -> draw x
  | Nx_dtype.Float8_e5m2 -> draw x
  | _ -> Nx.zeros (Nx.dtype x) shape
