(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [limits dt] is [(tiny, huge, eps)]. *)
let limits dt =
  let f = Nx.Dtype.float_format dt in
  (f.min_normal, f.max_finite, f.epsilon)

let tiny dt =
  let t, _, _ = limits dt in
  t

let huge dt =
  let _, h, _ = limits dt in
  h

let eps dt =
  let _, _, e = limits dt in
  e
