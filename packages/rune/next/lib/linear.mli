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

    Raises [Invalid_argument], at the operation, naming [t]'s entry point, if
    the operation is not linear in its slots, as in
    ["Rune.grad: a custom_jvp tangent map applies exp to a tangent; a tangent
     map must be linear in its tangents"], or if it reads a slot's value. *)

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
