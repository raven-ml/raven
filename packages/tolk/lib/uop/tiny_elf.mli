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

type field = { argument : argument; offset : int; size : int }
(** A field in a packed argument structure. *)

val layout : argument list -> field list
(** [layout signature] places arguments in declaration order, naturally aligned
    to their storage widths. Buffer addresses occupy eight bytes; scalar values
    occupy their dtype's width. Offsets start at zero.

    Raises [Invalid_argument] for void or weak scalar dtypes, or if offsets
    exceed the host size range. *)

val pack : field list -> bufs:nativeint array -> vals:int64 array -> bytes
(** [pack fields ~bufs ~vals] encodes a {!layout} in little-endian order,
    selecting addresses and scalar bit patterns by each argument's compact
    slot. Scalar values are narrowed to the declared width; padding is zero.
    [bufs] and [vals] must have the signature's buffer and scalar counts.

    Raises [Invalid_argument] if argument counts differ, a slot is outside
    the supplied arrays, or a field has an unsupported width. *)
