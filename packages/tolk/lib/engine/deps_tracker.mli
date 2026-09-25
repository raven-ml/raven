(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Byte-interval dependencies shared by queue compilation and execution. *)

type allocation = Graph of int | Parameter of int | Storage of int

type region = { base : allocation; lane : int option; start : int; stop : int }
(** A half-open byte interval of one allocation and optional device lane. *)

type 'a t
(** Read and write history carrying caller-defined submission tokens. *)

val create : unit -> 'a t
(** [create ()] is an empty dependency history. *)

val access : 'a t -> region list -> writes:int list -> 'a -> 'a list
(** [access t regions ~writes token] records one submission and returns its
    preceding read-after-write, write-after-read and write-after-write
    dependencies. [writes] indexes the written regions. Writes supersede only
    overlapping bytes of earlier accesses. Results may contain duplicates. *)

val buffer : Device.Buffer.t -> region
(** [buffer b] describes [b]'s byte interval within its root allocation. *)

val uop : Tolk_uop.Uop.t -> region
(** [uop u] describes a storage argument before allocation, preserving its
    base identity, device lane and contiguous byte views. Ordinary parameters
    identify an input slot, independently of their view shape. *)
