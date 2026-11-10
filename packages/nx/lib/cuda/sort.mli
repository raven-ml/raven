(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sorts and numberings: their launches of sort.cu.

    A plan takes a call nx.cuda found nothing to refuse in and chooses its
    path by the slice's length and the elements it keeps (sort.cu states
    the paths), then writes each launch's parameters. *)

val sort :
  axis:int ->
  descending:bool ->
  values:('v, 's) Nx_array.t ->
  positions:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  Bytes.t Launch.plan
(** [sort ~axis ~descending ~values ~positions x] plans
    {!Nx_kernel.S.sort}: one call, its state between its launches in the
    workspace. It is [Error Declined] where a launch would take more than
    [2{^31} - 1] groups. *)

val unique :
  ids:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  first:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  counts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  count:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  Bytes.t Launch.plan
(** [unique ~ids ~first ~counts ~count keys] plans {!Nx_kernel.S.unique}:
    a sort of the rows a word, then the numbering, as one call over the
    workspace, or, for keys of more than 8 words, as several of up to 8
    words' sorts each over a buffer of the call's own, so that the
    submissions a device keeps stay few. It is [Error Declined] as {!sort}
    is. *)

val set_run_words : int -> unit
(** [set_run_words n] makes a numbering's submission sort at most [n]
    words, at least one and at most 8, for tests. *)
