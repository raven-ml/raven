(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type argument = {
  name : string option;
  slot : int;
  dtype : Dtype.t;
  shape : int list;
  addrspace : Dtype.addr_space;
}

type t = {
  lib : bytes;
  name : string;
  target : Target.t;
  signature : argument list;
  profile_key : string option;
}

type field = { argument : argument; offset : int; size : int }

let layout signature =
  let offset = ref 0 in
  List.map (fun argument ->
      let size = if argument.addrspace = Dtype.Alu then Dtype.itemsize argument.dtype else 8 in
      if not (List.mem size [ 1; 2; 4; 8 ]) then
        invalid_arg "Tiny_elf.layout: arguments require a concrete scalar dtype";
      if !offset > max_int - (2 * size - 1) then
        invalid_arg "Tiny_elf.layout: argument layout exceeds the host size range";
      let aligned = (!offset + size - 1) / size * size in
      offset := aligned + size;
      { argument; offset = aligned; size }) signature

let pack fields ~bufs ~vals =
  let size, nbufs, nvals =
    List.fold_left
      (fun (size, nbufs, nvals) field ->
        let size = max size (field.offset + field.size) in
        if field.argument.addrspace = Dtype.Alu then size, nbufs, nvals + 1
        else size, nbufs + 1, nvals)
      (0, 0, 0) fields
  in
  if Array.length bufs <> nbufs || Array.length vals <> nvals then
    invalid_arg "Tiny_elf.pack: argument counts do not match the signature";
  let bytes = Bytes.make size '\000' in
  List.iter (fun field ->
      let arg = field.argument in
      let bits = if arg.addrspace = Dtype.Alu then
          vals.(arg.slot - nbufs)
        else Int64.of_nativeint bufs.(arg.slot) in
      match field.size with
      | 1 -> Bytes.set_uint8 bytes field.offset (Int64.to_int (Int64.logand bits 0xffL))
      | 2 -> Bytes.set_uint16_le bytes field.offset (Int64.to_int (Int64.logand bits 0xffffL))
      | 4 -> Bytes.set_int32_le bytes field.offset (Int64.to_int32 bits)
      | 8 -> Bytes.set_int64_le bytes field.offset bits
      | _ -> invalid_arg "Tiny_elf.pack: unsupported argument width") fields;
  bytes
