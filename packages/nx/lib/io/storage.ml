(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tensors and the host buffers that hold their elements *)

module B = Nx_device.Buffer

let create dtype n = B.create Nx_device.host (Nx_dtype.Scalar.of_dtype dtype) n

let tensor dtype buffer shape =
  Nx.reshape shape
    (Nx_effect.from_host Nx_effect.host_tensor_context dtype buffer)

(* The elements of [t] in C order: its own storage when [t] is contiguous. *)
let of_tensor t =
  let t = Nx.contiguous t in
  let b = Nx_effect.to_host t in
  B.view b ~offset:0 (B.dtype b) (Nx.numel t)

let bytes b = B.bigarray Bigarray.int8_unsigned b
