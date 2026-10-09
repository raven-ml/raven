(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Prepared submissions and {!Rig.submit}: the implementation of
    {!Rig.Submission}, with points as the ints of {!Point}.

    A submission's prepared form is C memory that a custom block owns and frees
    once collected, fixed once made. What one submit collects and answers is in
    the caller's run, so any number of submits of one submission run at once,
    each with its own run. *)

open Def

type t
type ref = { at : int; slot : int }

type work =
  | Words of buffer
  | Fill of {
      fill : nativeint;
      arg : buffer;
      ring_units : int;
      segment_bytes : int;
    }
  | Copy of { src : buffer; dst : buffer }
  | Launch of { image : image; kernel : string; params : int; refs : ref array }

type part = { queue : string; after : int array; work : work }

val make :
  ?hold:hold ->
  ?fixed:(buffer * access) list ->
  reads:int ->
  writes:int ->
  device ->
  part array ->
  t

type block = private int

val block : t -> int -> block

module Run : sig
  type t

  val make : unit -> t

  external groups :
    t ->
    (block[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"

  external threads :
    t ->
    (block[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"

  external shared : t -> (block[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_shared_byte" "caml_rig_run_shared"

  external int32 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int32_byte" "caml_rig_run_int32"

  external int64 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"

  external float32 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"

  external float64 :
    t -> (block[@untagged]) -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
end

val submit :
  t ->
  run:Run.t ->
  reads:buffer array ->
  writes:buffer array ->
  waits:int array ->
  int

val copy : device -> string -> src:buffer -> dst:buffer -> int
(** [copy d queue ~src ~dst] submits a copy of [src] into [dst], memory of [d],
    on [d]'s copy queue [queue], and is its point. It raises as
    {!Rig.Submission.make} and {!Rig.submit}. *)
