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

(* The bytes of the file at [path], read where they lie: the disk's mapping of
   its pages. Raises [Sys_error] if the file cannot be opened or mapped. *)
let file_bytes path =
  match Result.bind (B.of_file path) (B.borrow Nx_device.host) with
  | Ok b -> bytes b
  | Error why -> raise (Sys_error why)
