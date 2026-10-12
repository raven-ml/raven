(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Values' elements in host buffers, and host buffers as bytes *)

module B = Rig.Buffer

(* [claiming buffers f] is [f ()] with [buffers] under read claims, so that no
   compiled call lends their memory while [f] reads them. *)
let claiming buffers f = Rig.Claim.with_ ~read:buffers ~donate:[] (fun _ -> f ())

(* [reading ~by x f] is [f b] for [b] the elements of [x] in C order in a host
   buffer, read by the function [by]: its storage when they are one run of it on
   the host, under a read claim while [f] runs. *)
let reading ~by x f =
  let b = Nx.Op.eval (Read { by; x }) in
  claiming [ b ] (fun () -> f b)

(* [read_bytes b f] is [f] of the bytes of [b], once its last write is done,
   and [write_bytes b f] once every use of it is done. *)
let read_bytes b f = B.read b Bigarray.int8_unsigned f
let write_bytes b f = B.write b Bigarray.int8_unsigned f

(* The bytes of the file at [path], where they lie: the host's borrow of its
   pages. Raises [Sys_error] if the file cannot be opened or mapped. *)
let file path =
  match Rig_disk.of_file path with
  | Error why -> raise (Sys_error why)
  | Ok f -> (
      match B.borrow Rig.host f with
      | Some b -> b
      | None -> raise (Sys_error (path ^ ": the system does not map it")))

(* [mapped file kind shape ~off ~len] is the entry of [len] bytes at byte [off]
   of [file]: a value on the disk over them, or on a big-endian host their
   elements read and put in the host's byte order. *)
let mapped (type a b) file (kind : (a, b) Nx_dtype.t) shape ~off ~len =
  let size = Nx_dtype.itemsize kind in
  let view = B.view file ~first:off ~length:len in
  if Sys.big_endian then begin
    let n = len / size in
    let buffer = Nx_array.Elements.create kind n in
    B.copy ~src:view ~dst:buffer;
    write_bytes buffer (fun bytes ->
        Nx_io_codec.byteswap bytes ~element_size:size ~elements:n);
    Nx.P (Nx.of_buffer kind shape buffer)
  end
  else Nx.P (Nx.of_buffer kind shape view)
