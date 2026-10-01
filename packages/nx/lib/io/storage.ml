(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values' elements in host buffers, and host buffers as bytes *)

module B = Nx_device.Buffer

(* [claiming buffers f] is [f ()] with [buffers] under read claims, so that no
   compiled call lends their memory while [f] reads them. *)
let claiming buffers f =
  let rec claim = function
    | [] -> ()
    | b :: rest -> (
        B.Claim.read b;
        match claim rest with
        | () -> ()
        | exception e ->
            B.Claim.release b;
            raise e)
  in
  claim buffers;
  Fun.protect ~finally:(fun () -> List.iter B.Claim.release buffers) f

(* [reading ~by x f] is [f b] for [b] the elements of [x] in C order in a host
   buffer, read by the function [by]: its storage when they are one run of it on
   the host, under a read claim while [f] runs. *)
let reading ~by x f =
  let b = Nx.Op.eval (Read { by; x }) in
  claiming [ b ] (fun () -> f b)

let bytes b = B.bigarray Bigarray.int8_unsigned b

(* The bytes of the file at [path], read where they lie: the disk's mapping of
   its pages. Raises [Sys_error] if the file cannot be opened or mapped. *)
let file_bytes path =
  match Result.bind (B.of_file path) (B.borrow Nx_device.host) with
  | Ok b -> bytes b
  | Error why -> raise (Sys_error why)

(* [mapped file kind shape ~off ~len] is the entry of [len] bytes at byte [off]
   of [file]: a value on the disk over them, or on a big-endian host their
   elements read and put in the host's byte order. *)
let mapped (type a b) file (kind : (a, b) Nx_dtype.t) shape ~off ~len =
  let size = Nx_dtype.itemsize kind in
  let n = len / size in
  let view = B.view file ~offset:off (Nx_dtype.Scalar.of_dtype kind) n in
  if Sys.big_endian then begin
    let buffer = Nx_array.Elements.create kind n in
    B.copy ~src:view ~dst:buffer;
    Nx_io_codec.byteswap (bytes buffer) ~element_size:size ~elements:n;
    Nx.P (Nx.of_buffer kind shape buffer)
  end
  else Nx.P (Nx.of_buffer kind shape view)
