(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Elements of host buffers as values of a dtype.

    The elements of a buffer on {!Nx_device.host} are read and written through
    {!Nx_device.Buffer.bigarray} at their format's storage kind. Formats with no
    kind of their own go through {!Nx_dtype.Scalar.encode} and
    {!Nx_dtype.Scalar.decode}: bfloat16 and the float8 formats as their bits,
    [bool] as bytes [0] and [1], [bit] eight to a byte and 4-bit integers two
    to a byte, the first element in the low bits.

    Access is outside the devices' ordering, as {!Nx_device.Buffer.bigarray}'s
    is. *)

val create : ('a, 'b) Nx_dtype.t -> int -> Nx_device.Buffer.t
(** [create dt n] is a new buffer on {!Nx_device.host} of [n] elements of [dt],
    their contents unspecified.

    Raises as {!Nx_device.Buffer.create} does. *)

val get : ('a, 'b) Nx_dtype.t -> Nx_device.Buffer.t -> int -> 'a
(** [get dt b] reads the elements of [b] as values of [dt]: [get dt b i] is
    element [i]. The application to [dt] and [b] makes [b]'s view once, so apply
    it outside a loop over the elements.

    Raises [Invalid_argument] if [b] is not on {!Nx_device.host} or its format
    is not [Nx_dtype.Scalar.of_dtype dt]; the reader raises [Invalid_argument]
    if [i] is not in \[[0];[Nx_device.Buffer.length b - 1]\]. *)

val set : ('a, 'b) Nx_dtype.t -> Nx_device.Buffer.t -> int -> 'a -> unit
(** [set dt b] writes the elements of [b] as values of [dt]: [set dt b i v]
    stores [v] as element [i], rounding it as a store of [dt] does. An integer
    outside [dt]'s range keeps its low bits, at every width. It is applied as
    {!get} is, and raises as it does. *)

val fill : ('a, 'b) Nx_dtype.t -> Nx_device.Buffer.t -> 'a -> unit
(** [fill dt b v] stores [v] as every element of [b].

    Raises [Invalid_argument] as {!get} does. *)

val gather : Nx_device.Buffer.t -> View.t -> Nx_device.Buffer.t
(** [gather b v] is a new host buffer of the elements of the view [v] of [b], in
    C order, with their bits unchanged: a float's NaN payload is kept.

    Raises [Invalid_argument] if [b] is not on {!Nx_device.host} or if [v]
    reaches an element outside [b]. *)

val contiguous : Nx_device.Buffer.t -> View.t -> Nx_device.Buffer.t
(** [contiguous b v] is the elements of the view [v] of [b] in C order: a view
    of [b]'s own memory when [v] is C-contiguous and its first element starts a
    byte, {!gather}[ b v] otherwise.

    Raises [Invalid_argument] as {!gather} does. *)
