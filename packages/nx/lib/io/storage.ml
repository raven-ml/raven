(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values' elements in host buffers, and host buffers as bytes *)

module B = Nx_device.Buffer

(* The elements of [x] in C order, in a host buffer, read by the function [by]:
   its storage when they are one run of it on the host. *)
let elements ~by x = Nx.Op.eval (Read { by; x })
let bytes b = B.bigarray Bigarray.int8_unsigned b

(* The bytes of the file at [path], read where they lie: the disk's mapping of
   its pages. Raises [Sys_error] if the file cannot be opened or mapped. *)
let file_bytes path =
  match Result.bind (B.of_file path) (B.borrow Nx_device.host) with
  | Ok b -> bytes b
  | Error why -> raise (Sys_error why)
