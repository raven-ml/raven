(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thrift's compact protocol, read from byte arrays.

    Parquet's footer and page headers are Thrift structures in the
    {{:https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md}compact
     protocol}. A reader decodes values from a range of a byte array. A decoder
    of a structure iterates its fields with {!fields} and reads each value with
    the function of its type, or {!skip}s it, so that the fields talon does not
    read, including those of later versions of the format, are skipped.

    Every function raises {!Error} at the first byte that does not decode: past
    the reader's limit, a value of another wire type than the one asked for, a
    varint longer than its type, an [i32] outside 32 bits, an [i64] outside
    OCaml's [int], a list longer than the bytes left, or values nested deeper
    than 64 levels. The product of two [i32]s therefore fits an [int]. *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

exception Error of int * string
(** [Error (pos, msg)] is raised on the byte at [pos] of the array, which does
    not decode for the reason [msg]. *)

(** {1:readers Readers} *)

type t
(** The type for readers: a byte array, the position of the next byte to read
    and a limit. *)

val make : bigbytes -> pos:int -> limit:int -> t
(** [make b ~pos ~limit] reads [b] from [pos], up to the byte before [limit].

    Raises [Invalid_argument] if [0 <= pos <= limit <= Bigarray.Array1.dim b]
    does not hold. *)

val pos : t -> int
(** [pos r] is the position of the next byte [r] reads. *)

(** {1:values Values} *)

type ty
(** The type for the wire types of values: Thrift's boolean, integers, double,
    binary, list, set, map and structure. *)

val fields : t -> (int -> ty -> unit) -> unit
(** [fields r f] reads the fields of a structure, up to its stop field, calling
    [f id ty] on each field's identifier and wire type. [f] must read the
    field's value with the function of its type, or {!skip} it. *)

val structure : t -> ty -> (int -> ty -> unit) -> unit
(** [structure r ty f] is [fields r f] for the value of a field whose wire type
    is [ty], which must be a structure. *)

val bool : t -> ty -> bool
(** [bool r ty] reads a boolean, which a field's wire type holds and a list's
    element is a byte for. *)

val i8 : t -> ty -> int
(** [i8 r ty] reads an [i8]. *)

val i32 : t -> ty -> int
(** [i32 r ty] reads an [i32]. *)

val i64 : t -> ty -> int
(** [i64 r ty] reads an [i64]. *)

val binary : t -> ty -> string
(** [binary r ty] reads a binary value, which Parquet uses for strings. *)

val list : t -> ty -> (t -> ty -> 'a) -> 'a list
(** [list r ty elt] reads a list or a set, each element with [elt r ty'] for the
    elements' wire type [ty']. *)

val skip : t -> ty -> unit
(** [skip r ty] reads a value of wire type [ty] and drops it. *)
