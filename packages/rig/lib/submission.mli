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
    made. The prepared form names stamps, and each launch's [launch] from
    its image ({!Rig_edge.type-entry}), without a reference, so the record
    beside it keeps their owners reachable: the buffers of its words and
    copy parts and its fixed memory, the images of its launches and entry
    refs, and the token of its keep. It also holds each entry ref's function
    value and place in a run's blocks. A copy between two buffers is set
    once, at make.

    A submit's state, what it collects and answers, is a {e call}
    ([struct rig_call] in [rig_stubs.h]): C memory that one submit at a time
    takes from every call made, and that is never freed, so their number is
    the most submits that ever ran at once, and a warm submit allocates no
    call. A submit goes like this:
    + In one C call, it moves the caller's run to submitting, from the
      calling thread's taking or from free, unless the run holds no bytes,
      and takes a free call, fitted to the submission unless it served it
      last.
    + It takes a read claim on each fixed buffer and each of the run's
      buffers, checks it live, and names it in the call's slot. For a slot
      that a copy names, it also records the buffer's side of the copy
      ([struct rig_side] in [rig_stubs.h]): the handle, offset and length
      the copy passes the driver, the span and alias its overlap check
      compares, and whether the side is this process's memory and the
      device's own memory.
    + It collects, per other device, the greatest point the work follows
      ([caml_rig_call_collect]), with the caller's waits, and writes each
      entry ref's value into the run's blocks. A submission with a copy that
      has a slot end hands its driver the call's duplicate of its parts. The
      call sets each such copy's fields from its buffers once it has checked
      their lengths and overlap.
    + For each point not done, it waits on the host, or commits the
      producer's work and adds a wait in the device's queue. The device waits
      in its queue for a producer it can read ({!Def.live}'s [pairs]), up
      to the [most] waits its driver takes in one call of its [submit].
    + In one C call that runs no OCaml, it waits while another call is
      submitting or committing on the device, calls the driver's [room],
      assigns the next value, calls the driver's [submit] and raises the
      stamps.
    + It releases the claims, gives the call back and frees the run.

    What that C call returns ({!Dev.Answer}) makes the submit retry while
    another call holds the device, wait for the device's next value while its
    queues have no room, grow the device's record of waits in its queue, or
    raise. What one submit collects and returns is in its call, so submits
    of one submission run concurrently as {!Rig.Submission} and
    {!Rig.Submission.Run.take} state.

    While a profile is being taken, a submit asks its driver for the times of
    its value and reads, from its buffers, the devices and bytes of each copy
    part. It leaves each profile an entry ({!Profile.pending}) that the
    profile's {!Profile.take} reads once the value is reached: its [Work]
    and a copy per copy part, with the driver's times, or, where the driver
    wrote none, from the submit to that read. *)

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

val entry : func -> at:int -> ref
(** [entry f ~at] is {!Rig.Submission.entry}[ f ~at]. *)

val copy : src:side -> dst:side -> part
(** [copy ~src ~dst] is {!Rig.Submission.copy}[ ~src ~dst]. *)

val set : dst:side -> char -> part
(** [set ~dst c] is a copy part that sets every byte of [dst] to [c],
    reading no memory: a fill at the edge ([copy.fill] in [rig_edge.h]). It
    checks nothing of [dst]; {!make} refuses a [dst] buffer that is dead,
    not the device's, or this process's memory. *)

val launch : ?refs:ref array -> func -> params:int -> part
(** [launch ~refs f ~params] is {!Rig.Submission.launch}[ ~refs f ~params]. *)

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

(** Runs: the blocks of one submission, C memory that a custom block owns,
    sized when made, and a state word that says who holds them: free,
    submitting, or the thread that took them. Each setter is an external
    that takes untagged ints and unboxed floats, so no argument is boxed at a
    call, inlined or not. *)
module Run : sig
  type submission := t

  type t
  (** {!Rig.Submission.Run.t}. *)

  val make : submission -> t
  (** [make s] is {!Rig.Submission.Run.make}[ s]. *)

  val submission : t -> submission
  (** [submission r] is {!Rig.Submission.Run.submission}[ r]. *)

  external take : t -> bool = "caml_rig_run_take" [@@noalloc]
  (** {!Rig.Submission.Run.take}. *)

  external release : t -> unit = "caml_rig_run_release" [@@noalloc]
  (** {!Rig.Submission.Run.release}. *)

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

  external int64 :
    t -> block -> (int[@untagged]) -> (int64[@unboxed]) -> unit
    = "caml_rig_run_int64_byte" "caml_rig_run_int64"
  (** {!Rig.Submission.Run.int64}. *)

  external float32 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float32_byte" "caml_rig_run_float32"
  (** {!Rig.Submission.Run.float32}. *)

  external float64 : t -> block -> (int[@untagged]) -> (float[@unboxed]) -> unit
    = "caml_rig_run_float64_byte" "caml_rig_run_float64"
  (** {!Rig.Submission.Run.float64}. *)
end

val submit : ?waits:int array -> Run.t -> buffers:buffer array -> int
(** [submit ~waits r ~buffers] is {!Rig.submit}[ ~waits r ~buffers]. *)

val run_copy : device -> waits:int array -> src:buffer -> dst:buffer -> int
(** [run_copy d ~waits ~src ~dst] submits a copy of [src] into [dst] on [d]'s
    copy engine, after the points of [waits], and is its point. It submits
    [d]'s submission of the one part [copy ~src:(Slot 0) ~dst:(Slot 1)], made
    once per device with its run, which holds no bytes, so any number of
    copies submit it at once. [d] must reach both buffers. They are [d]'s,
    or, for a device of another machine, one of them may be this process's
    memory. It raises as {!Rig.Submission.make} and {!Rig.submit} do. *)

val run_boxes :
  device -> waits:int array -> src:buffer -> dst:buffer -> box array -> int
(** [run_boxes d ~waits ~src ~dst boxes] is {!run_copy} of the rows of each
    box of [boxes], in order, from [src] into [dst]. It makes [d]'s
    submission of one part per box for the call. Each box must fit [src]
    and [dst] ({!Rig.Buffer.type-box}), which nothing checks. *)

val run_set : device -> waits:int array -> dst:buffer -> char -> int
(** [run_set d ~waits ~dst c] submits a fill of [dst] with [c] on [d]'s copy
    engine, after the points of [waits], and is its point. It submits [d]'s
    submission of the one part [set ~dst:(Slot 0) c], made with its run at
    [c]'s first fill on [d], which holds no bytes, so any number of fills
    submit it at once. [dst] must be [d]'s. It raises as
    {!Rig.Submission.make} and {!Rig.submit} do. *)

val mark : unit -> int
(** [mark ()] submits the host's submission of no part, made once, and is
    its point: the next value of the host's timeline, reached when it
    returns. *)
