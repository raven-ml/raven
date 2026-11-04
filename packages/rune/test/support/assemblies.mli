(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Assemblies ({!Nx.Prim.Assemble}) of random bits: up to three pieces at
    regions of a drawn shape, forwards, backwards or by steps, empty ones among
    them, after a first piece that holds every element; or many pieces of one
    element, or as many rows of a stack. Each piece is a view of an argument
    or a value the program computes from two. *)

type t
(** The type for cases. *)

val pp : Format.formatter -> t -> unit

val dtype : t -> Nx.Dtype.any
(** [dtype c] is [c]'s dtype. *)

val gen : t Windtrap.Gen.t
(** [gen] draws cases of every dtype, half of them few pieces at random regions
    of a drawn shape, half many pieces, from 2 to 64. *)

val examples : t list
(** [examples] holds a case of each dtype, in turn a stack of 33 rows, 22
    single elements into 64, one piece by steps backwards whose steps reach
    past its axis, no piece, one view over a computed piece that covers the
    value, pieces that overlap, and one computed piece over the fill. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that the assembly compiled on [s]'s first device, its
    arguments placed there, gives eager's bits on the host, with no call
    evaluated. It covers no piece, one view over the fill and over a piece that
    holds every element, one computed piece, several pieces, pieces that tile
    and that overlap, regions by steps and backwards, a region whose steps
    reach past its axis, 33 pieces or more, and complex and sub-byte
    dtypes. *)
