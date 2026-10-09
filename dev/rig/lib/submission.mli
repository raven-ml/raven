(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Prepared submissions and {!Rig.submit}: the implementation of
    {!Rig.Submission}, with points as the ints of {!Point}.

    A submission's prepared form is C memory that a custom block owns and frees
    once collected. A submit holds the submission's guard throughout, so two
    domains' submits of one submission take turns. *)

open Def

type t

type work =
  | Words of buffer
  | Fill of {
      fill : nativeint;
      arg : buffer;
      ring_units : int;
      segment_bytes : int;
    }
  | Copy of { src : buffer; dst : buffer }

type part = { queue : string; after : int array; work : work }

val make : ?hold:hold -> reads:int -> writes:int -> device -> part array -> t

val submit :
  t -> reads:buffer array -> writes:buffer array -> waits:int array -> int

val copy : device -> string -> src:buffer -> dst:buffer -> int
(** [copy d queue ~src ~dst] submits a copy of [src] into [dst], memory of [d],
    on [d]'s copy queue [queue], and is its point. It raises as
    {!Rig.Submission.make} and {!Rig.submit}. *)
