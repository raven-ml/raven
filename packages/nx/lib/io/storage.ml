(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensors and the host buffers that hold their elements *)

module B = Nx_device.Buffer

let tensor dtype buffer shape =
  Nx.reshape shape
    (Nx_effect.from_host Nx_effect.Placement.host dtype buffer)

let bytes b = B.bigarray Bigarray.int8_unsigned b
