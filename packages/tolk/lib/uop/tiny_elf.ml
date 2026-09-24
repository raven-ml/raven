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
