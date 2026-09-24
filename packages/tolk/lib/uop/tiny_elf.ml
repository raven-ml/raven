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
