(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sorts and numberings, planned as calls of the [sort_*] and [unique_*]
    kernels. *)

val sort :
  axis:int ->
  descending:bool ->
  values:('v, 's) Nx_array.t ->
  positions:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [sort ~axis ~descending ~values ~positions x] plans
    {!Nx_kernel.S.sort} of a call that fits: [None] where it has no
    element, else a block sort in LDS of slices of at most 4096 elements
    (2048 of complex128), a select then a block sort where each slice
    keeps at most a quarter of that, and otherwise a radix sort of a pass
    a key byte. Its workspace and tickets are those nx_amd.mli states, and
    no launch's groups lie in the slow band ({!Plan.in_band}). *)

val unique :
  ids:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  first:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  counts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  count:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [unique ~ids ~first ~counts ~count keys] plans {!Nx_kernel.S.unique}
    of a call that fits: a radix sort of the rows a word at a time from
    the last, then their numbering. No launch's groups lie in the slow
    band ({!Plan.in_band}). *)

val bound_sorts : parts:int -> groups:int -> unit
(** [bound_sorts ~parts ~groups] makes a radix pass cut a slice into at
    most [parts] parts, in place of 256, and a sort's launches take at
    most [groups] groups, in place of {!Plan.stride_groups}, so that tests
    reach parts of several tiles and groups of several turns with small
    arrays. It applies to calls planned after it returns.

    Raises [Invalid_argument] if [parts] is not from 1 to 256 or [groups]
    not from 1 to {!Plan.stride_groups}. *)
