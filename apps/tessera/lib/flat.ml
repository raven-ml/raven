(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 's) t = ('v, 's, Bigarray.c_layout) Bigarray.Array1.t

let read = Nx.Host.read

let create k n v =
  let b = Bigarray.Array1.create k Bigarray.c_layout n in
  Bigarray.Array1.fill b v;
  b

let wrap (_ : ('v, 's) Nx.dtype) shape (b : ('v, 's) t) = Nx.Host.wrap shape b
