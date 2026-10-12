(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Elements of buffers as values of a dtype.

    A buffer holds bytes ({!Rig.Buffer.t}); a dtype reads them as elements.
    Each element takes {!Nx_dtype.Scalar.bitsize} bits of its dtype's format,
    and [n] elements take {!bytes}[ dt n] bytes. Formats with no bigarray kind
    of their own go through {!Nx_dtype.Scalar.encode} and
    {!Nx_dtype.Scalar.decode}: bfloat16 and the float8 formats as their bits,
    [bool] as bytes [0] and [1], [bit] eight to a byte and 4-bit integers two
    to a byte, the first element in the low bits.

    {!read}, {!write}, {!fill}, {!gather} and {!contiguous} access the bytes
    in place, so the host must address the buffer's memory, as on
    {!Rig.host}. They are ordered as {!Rig.Buffer.read} and
    {!Rig.Buffer.write} order host access, and raise as those do. *)

val bytes : ('a, 'b) Nx_dtype.t -> int -> int
(** [bytes dt n] is the number of bytes [n] elements of [dt] take, the last
    byte whole. *)

val length : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> int
(** [length dt b] is the number of elements of [dt] that [b]'s bytes hold. *)

val create : ('a, 'b) Nx_dtype.t -> int -> Rig.Buffer.t
(** [create dt n] is a new buffer on {!Rig.host} of [n] elements of [dt],
    their contents unspecified. The bits of its last byte past its last
    element, for a dtype of fewer than 8 bits, are [0].

    Raises as {!Rig.Buffer.create} does. *)

val read : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> ((int -> 'a) -> 'c) -> 'c
(** [read dt b f] is [f get], where [get i] is element [i] of [b] as a value
    of [dt], once the last write of [b]'s memory is done. [get] is valid only
    during [f].

    [get] raises [Invalid_argument] if [i] is not in
    \[[0];[length dt b - 1]\]. *)

val write :
  ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> ((int -> 'a -> unit) -> 'c) -> 'c
(** [write dt b f] is [f set], where [set i v] stores [v] as element [i] of
    [b], rounding it as a store of [dt] does, once every use of [b]'s memory
    is done. An integer outside [dt]'s range keeps its low bits, at every
    width. [set] is valid only during [f], and raises as {!read}'s [get]
    does. *)

val fill : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> 'a -> unit
(** [fill dt b v] stores [v] as every element of [b]. *)

val gather : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> View.t -> Rig.Buffer.t
(** [gather dt b v] is a new host buffer of the elements of [dt] at the view
    [v] of [b], in C order, with their bits unchanged: a float's NaN payload
    is kept.

    Raises [Invalid_argument] if [v] reaches an element outside [b]. *)

val contiguous : ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> View.t -> Rig.Buffer.t
(** [contiguous dt b v] is the elements of [dt] at the view [v] of [b] in C
    order: a view of [b]'s own memory when [v] is C-contiguous and its first
    element starts a byte, {!gather}[ dt b v] otherwise.

    Raises [Invalid_argument] as {!gather} does. *)
