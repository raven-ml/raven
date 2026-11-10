(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 's) t = ('v, 's, Bigarray.c_layout) Bigarray.Array1.t

let read x f =
  let n = Nx.numel x in
  let flat a = Nx_array.read a (fun g -> f (Bigarray.reshape_1 g n)) in
  Nx.Repr.read x (fun a ->
      if Nx_array.Layout.is_contiguous (Nx_array.layout a) then flat a
      else flat (Nx_array.copy a))

let create k n v =
  let b = Bigarray.Array1.create k Bigarray.c_layout n in
  Bigarray.Array1.fill b v;
  b

let wrap dt shape b =
  let g = Bigarray.reshape (Bigarray.genarray_of_array1 b) shape in
  Nx.Repr.unsafe_of_array Nx.Host.v (Nx_array.of_bigarray dt g)
