(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Rank lifting and per-channel correlation shared by the image operations.

    The operations work on batches [[N; H; W; C]]; {!with_batch} and
    {!with_batch_pair} let them take a single image [[H; W; C]] too. *)

val err_rank : int -> string
(** [err_rank n] is the error message for an image of rank [n], which is
    neither 3 nor 4. *)

val with_batch :
  (('a, 'b, 'd) Nx.t -> ('c, 'e, 'd) Nx.t) ->
  ('a, 'b, 'd) Nx.t ->
  ('c, 'e, 'd) Nx.t
(** [with_batch f img] is [f img] for a batch. For a single image it adds a
    batch axis of size 1 before [f] and removes it after.

    Raises [Invalid_argument] if [img]'s rank is neither 3 nor 4. *)

val with_batch_pair :
  (('a, 'b, 'd) Nx.t -> ('c, 'e, 'd) Nx.t * ('f, 'g, 'd) Nx.t) ->
  ('a, 'b, 'd) Nx.t ->
  ('c, 'e, 'd) Nx.t * ('f, 'g, 'd) Nx.t
(** [with_batch_pair f img] is {!with_batch} for an [f] that returns a pair. *)

val at : ('a, 'b, 'd) Nx.t -> 'd Nx.Placement.t
(** [at img] is whole on each device of [img]'s placement: where a kernel or a
    constant made for [img] lies. *)

val beside : ('a, 'b, 'd) Nx.t -> ('c, 'e, Nx.host) Nx.t -> ('c, 'e, 'd) Nx.t
(** [beside img k] is the host value [k], a kernel or a constant made from OCaml
    numbers, at {!at}[ img]. *)

val convolve_per_channel :
  ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [convolve_per_channel kernel img] correlates each channel of the batch
    [img] with [kernel] of shape [[kH; kW]], without flipping it. The result
    keeps [img]'s shape. Each pixel's window spans [kH / 2] rows and
    [kW / 2] columns before it, and pixels outside the image count as
    zero. *)
