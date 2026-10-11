(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Prepared submissions and {!Rig.submit}: the implementation of
    {!Rig.Submission}, with points as the ints of {!Point}.

    {!make} copies the caller's parts and checks them. It derives the run's
    slots from their refs and copies, and each part's queue and the parts it
    follows from the memory the parts use ({!Rig_edge.type-queue}). It
    compiles them into a prepared form ([struct rig_sub] in [rig_stubs.h]), C
    memory that a custom block owns and frees once collected, fixed once
    made. The
    prepared form names stamps, and each launch's [launch] from its image
    ({!Rig_edge.type-entry}), without a reference, so the record beside it
    keeps their owners reachable: the buffers of its words and copy parts
    and its fixed memory, the images of its launches and entry refs, and the
    hold of its keep. It also holds each entry ref's function value and place
    in a run's blocks.

    A submit goes like this:
    + It takes the caller's run by compare-and-set, and fits the run to the
      submission unless the run served it last.
    + It takes a read claim on each fixed buffer and each of the run's
      buffers, checks it live, and names it in the run's slot, with what a
      copy that names the slot passes the driver.
    + It collects, per other device, the greatest point the work follows
      ([caml_rig_run_collect]), with the caller's waits, and writes each
      entry ref's value into the run's blocks. A submission with copies hands
      its driver the run's copy of its parts, the fields of each copy with a
      slot end set from its buffers after their sizes and overlap are
      checked. A copy between two fixed buffers is set once, at make.
    + For each point not done, it waits on the host, or commits the
      producer's work and adds a wait in the device's queue. The device waits
      in its queue for a producer it can read ({!Def.live}'s [pairs]), up
      to the [most] waits its driver takes in one call of its [submit].
    + In one C call that runs no OCaml, it waits while another call is
      submitting or committing on the device, calls the driver's [room],
      assigns the next value, calls the driver's [submit] and raises the
      stamps.
    + It releases the claims and gives the run back.

    What that C call returns ({!Dev.Answer}) makes the submit retry while
    another call holds the device, wait for the device's next value while its
    queues have no room, grow the device's record of waits in its queue, or
    raise. What one submit collects and returns is in its run, so any number
    of submits of one submission run concurrently, each with its own run.

    While a profile is being taken, a submit asks its driver for the times of
    its value and records them as a span once a wait finds the value
    reached ({!Dev.after}). *)

open Def

type t
(** {!Rig.Submission.t}. *)

type ref
(** {!Rig.Submission.ref}. *)

type side = Slot of int | Buffer of buffer  (** {!Rig.Submission.side}. *)

type part
(** {!Rig.Submission.part}. *)

val read : int -> at:int -> ref
(** [read slot ~at] is {!Rig.Submission.read}[ slot ~at]. *)

val write : int -> at:int -> ref
(** [write slot ~at] is {!Rig.Submission.write}[ slot ~at]. *)

val unordered : int -> at:int -> ref
(** [unordered slot ~at] is {!Rig.Submission.unordered}[ slot ~at]. *)

val entry : image -> string -> at:int -> ref
(** [entry image f ~at] is {!Rig.Submission.entry}[ image f ~at]. *)

val copy : src:side -> dst:side -> part
(** [copy ~src ~dst] is {!Rig.Submission.copy}[ ~src ~dst]. *)

val launch : ?refs:ref array -> image -> string -> params:int -> part
(** [launch ~refs image f ~params] is {!Rig.Submission.launch}[ ~refs image f
    ~params]. *)

val fill : ?refs:ref array -> nativeint -> params:int -> part
(** [fill ~refs f ~params] is {!Rig.Submission.fill}[ ~refs f ~params]. *)

val words : buffer -> part
(** [words b] is {!Rig.Submission.words}[ b]. *)

val make :
  ?keep:'a -> ?fixed:(buffer * access) list -> device -> part array -> t
(** [make ?keep ?fixed d parts] is {!Rig.Submission.make}[ ?keep ?fixed d
    parts]. A change to [parts] while it runs, from another domain or a
    thread, changes nothing it makes. *)

type block
(** {!Rig.Submission.block}: the id of the submission's prepared form, where
    the block lies in a run with its parameter count, and the most threads a
    group of its function has. *)

val block : t -> int -> block
(** [block s i] is {!Rig.Submission.block}[ s i]. *)

val address : t -> buffer -> int
(** [address s b] is {!Rig.Submission.address}[ s b]. *)

(** Runs: C memory that a custom block owns, which one submit or one store
    holds at a time. Each setter is an external that takes untagged ints and
    unboxed floats, so no argument is boxed at a call, inlined or not. *)
module Run : sig
  type t
  (** {!Rig.Submission.Run.t}. *)

  val make : unit -> t
  (** [make ()] is {!Rig.Submission.Run.make}[ ()]. *)

  external groups :
    t ->
    block ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_groups_byte" "caml_rig_run_groups"
  (** {!Rig.Submission.Run.groups}. *)

  external threads :
    t ->
    block ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    (int[@untagged]) ->
    unit = "caml_rig_run_threads_byte" "caml_rig_run_threads"
  (** {!Rig.Submission.Run.threads}. *)

  external shared : t -> block -> (int[@untagged]) -> unit
    = "caml_rig_run_shared_byte" "caml_rig_run_shared"
  (** {!Rig.Submission.Run.shared}. *)

  external int32 : t -> block -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int32_byte" "caml_rig_run_int32"
  (** {!Rig.Submission.Run.int32}. *)

  external int64 : t -> block -> (int[@untagged]) -> (int[@untagged]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"
  (** {!Rig.Submission.Run.int64}. *)

  external float32 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"
  (** {!Rig.Submission.Run.float32}. *)

  external float64 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
  (** {!Rig.Submission.Run.float64}. *)
end

val submit : ?waits:int array -> t -> run:Run.t -> buffers:buffer array -> int
(** [submit ~waits s ~run ~buffers] is {!Rig.submit}[ ~waits s ~run
    ~buffers]. *)

val run_copy : device -> src:buffer -> dst:buffer -> int
(** [run_copy d ~src ~dst] submits a copy of [src] into [dst] on [d]'s copy
    engine, and is its point. It submits [d]'s submission of the one part
    [copy ~src:(Slot 0) ~dst:(Slot 1)], made once per device, with the
    calling domain's copy run, or a fresh run if another thread of the domain
    holds that one.
    [d] must reach both buffers: they are [d]'s, or, for a device of another
    machine, one side may be this process's memory. It raises as
    {!Rig.Submission.make} and {!Rig.submit} do. *)
