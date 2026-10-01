(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reverse mode: slots, the recorder and the transposes.

    A tape is one application of reverse mode. Its slots are traced values that
    stand for tangents: an input's, or the result of an operation linear in
    slots, which the tape's interpreter records instead of computing. An
    operation with no slot among its operands is forwarded and computed.
    {!transpose} walks the recording backwards, adding each entry's cotangent
    into its slot operands' entries. *)

type tape
(** The type for tapes. *)

val create : string -> tape
(** [create entry] is a fresh tape for the entry point named [entry], which its
    errors name. *)

val input : tape -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [input t x] is a fresh slot of [t] with [x]'s metadata, which nothing feeds.
*)

val install : tape -> (unit -> 'a) -> 'a
(** [install t f] is [f ()] with its operations recorded on [t] when a slot of
    [t] is an operand ({!Construct.install}).

    An operation is linear in its slots when every other operand is a
    coefficient: negation; addition and subtraction of two slots; a product with
    one slot; a quotient of a slot; a selection, a sum or running sum, a zero
    padding, a concatenation, a gather, a scatter or an update, a window or its
    fold, a Fourier transform, a copy, a movement or a placement of slots; a
    cast among real and complex dtypes; a matrix product with one slot; and a
    triangular solve for a slot. In a selection, a concatenation, a scatter and
    an update, an operand that is no slot stands for zero, as a tangent's zero
    fill does.

    Raises [Invalid_argument], at the operation, naming [t]'s entry point, if
    the operation is not linear in its slots, as in
    ["Rune.grad: a custom_jvp tangent map applies exp to a tangent; a tangent
     map must be linear in its tangents"], an addition or a nonzero padding
    being affine, or if it reads a slot's value. *)

val call :
  tape ->
  Nx.packed list ->
  (Nx.packed list -> Nx.packed list) ->
  Nx.packed list ->
  Nx.packed list
(** [call t xs pullback ys] records a linear map from the slots [xs] of [t] to
    fresh slots with the metadata of [ys], which it returns, one per value of
    [ys]. Its transpose is [pullback]: given the cotangents of the outputs, in
    order, zeros where none arrived, it returns the cotangents of [xs], in
    order. [pullback] runs only when an output received a cotangent. *)

(** {1:transposing Transposing} *)

type cotangents
(** The type for the cotangents of a tape's slots. *)

val cotangents : tape -> cotangents
(** [cotangents t] is no cotangent for any slot of [t]. *)

val add : cotangents -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit
(** [add cts x ct] adds [ct] to the cotangent of [x] if [x] is a slot of [cts]'s
    tape, and does nothing otherwise. *)

val transpose : cotangents -> unit
(** [transpose cts] adds to [cts] the transpose of each recorded entry applied
    to its cotangent, from the last entry to the first, skipping those with
    none. Its operations run in the caller's interpretation. *)

val cotangent : cotangents -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t option
(** [cotangent cts x] is the cotangent of the slot [x], if it received one. *)
