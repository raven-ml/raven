(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Compiled binaries and their argument signatures. *)

type argument = {
  name : string option;
  slot : int;
      (** Compact dispatch slot: buffers in globals order, followed by scalars.
          This is independent of the original call's parameter slots. *)
  dtype : Dtype.t;
      (** Element dtype for storage, value dtype for scalars. *)
  shape : int list;
      (** Maximum shape at compilation. Scalar values have an empty shape. *)
  addrspace : Dtype.addr_space;
      (** {!Dtype.Alu} denotes a scalar value; other spaces denote storage.
          A name or an empty shape alone does not distinguish them. *)
}

type t = {
  lib : bytes;
  name : string;
      (** Compiled entry point. *)
  target : Target.t;
  signature : argument list;
      (** Buffer formals in linear program order, followed by scalar formals
          in their binding order. *)
  profile_key : string option;
      (** Semantic identity of the compiled program, when available. *)
}
