(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dev keeps the table of open devices, and runs each device's timeline
    and loss.

    A device is four records. It takes a slot, its index, whose C line
    ([struct rig_line] in [rig_stubs.h]) holds its timeline and state, and
    whose word, in rig's array, lives for the process. The OCaml record
    ({!Def.device}) holds its facts, and its {!Def.live} its memory lists and
    what only its work needs. The C record ([struct rig_device]) holds its
    reason, its counted calls, its turn and the waits its queues hold on
    other devices. Its stop releases what the device needs only to run work.
    Once it has stopped and nothing of it remains to give back, its slot
    goes to a later device ({!free_slot}), whose timeline goes on from the
    slot's word ({!Def.device.base}), and an earlier device's points are
    answered from the slot's line. This module
    implements {!Rig}'s opening and closing
    ({!Rig.section-devices}), waits on a timeline ({!Rig.section-timeline})
    and loss ({!Rig.section-loss}).

    Any domain may call any function. A device's lock ({!protect}) guards the
    mutable fields of its {!Def.live}. No driver call and no wait runs under
    it but a lost device's read of a word of its driver's own ({!stopped})
    and a read of its next value ({!next_value}), which waits for its C
    mutex alone, and it is not reentrant. The host is never lost. The life
    of a device from its loss to its stop, and what a fork does, are in
    {!section-loss}. {!Rig.section-timeline} and {!Rig.section-loss} state
    what timelines and a loss mean to a caller. *)

open Def

exception Lost of device * string
(** {!Rig.Lost}. *)

exception Out_of_memory of device * int
(** {!Rig.Out_of_memory}. *)

val host : device
(** [host] is {!Rig.host}, of index [0]. *)

(** {1:locks Locks} *)

val protect : device -> ('a -> 'b) -> 'a -> 'b
(** [protect d f x] is {!Lock.protect} of [d]'s lock. *)

val check_held : string -> unit
(** [check_held fn] raises [Invalid_argument] naming [Rig.fn] if the calling
    thread holds a lock ([rig_holds] in [rig_edge.h]), as a call made inside
    a lock does ({!Rig.section-domains}). Every function of {!Rig} that may
    take a lock calls it first, before it changes anything, but {!Rig.fail},
    which reads [rig_holds] itself; [fn] is the function's name in {!Rig},
    such as ["Buffer.create"]. It allocates nothing unless it raises. *)

(** {1:table The table of devices} *)

val of_index : int -> device
(** [of_index i] is the last device that took the slot [i], {!host} for [0]
    or a slot no device took. *)

val owns : device -> int -> bool
(** [owns d p] is [true] iff [p] is a point of [d]: of [d]'s slot, and above
    the value [d]'s timeline started after. *)

val drain_all : (unit -> unit) ref
(** [drain_all] drains every device ({!Memory.drain_all}), which may free
    their slots. {!Memory} sets it. An open that finds every slot held
    collects and calls it once before it is [Error]. *)

val iter : (device -> unit) -> unit
(** [iter f] applies [f] to each published device other than the host, lost
    ones included, in index order. *)

val is_host : device -> bool
(** [is_host d] is [true] iff [d] is {!host}. *)

val is_io : device -> bool
(** [is_io d] is [true] iff [d] is an io device. *)

val same_machine : device -> device -> bool
(** [same_machine d d'] is [true] iff [d] and [d'] are devices of one
    machine. *)

val host_of : device -> device
(** [host_of d] is {!Rig.host_of}[ d]. *)

val machine_of : device option -> machine
(** [machine_of d] is the machine of a device opened beside [d], this one for
    [None]: {!Rig.open_}'s [machine]. *)

val reaches : device -> device -> bool
(** [reaches d d'] is {!Rig.reaches}[ d d']. *)

val open_driver :
  ?memory_device:bool ->
  (module Rig_edge.Driver with type t = 'a) ->
  machine:machine ->
  name:string ->
  (word:nativeint -> ('a, string) result) ->
  (device, string) result
(** [open_driver (module D) ~machine ~name make] is {!Rig.open_} on
    [machine], or {!Rig.open_host} for [Hosts]. [make] is given the device's
    slot's word ({!Rig_edge.type-opener}). Concurrent opens of one name
    wait for the first. It reads the device's facts before it puts the
    device in the table. A fault while reading them stops the handle and is
    [Error], and facts that break the driver's contract stop the handle and
    raise [Invalid_argument]. With [memory_device] (defaults to [false]),
    {!Rig.runs_on_host} is [true] of the device, and misuse raises under
    {!Rig.memory_device}'s name. *)

val open_io :
  (module Rig_edge.Io with type t = 'a) ->
  machine:machine ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
(** [open_io (module I) ~machine ~name make] is {!Rig.open_io} on
    [machine]. Concurrent opens of one name wait for the first. A fault
    while the device's budget is read stops the handle and is [Error]. *)

(** {1:local Values per device} *)

type 'a key
(** The type for {!Rig.Local} keys. *)

val new_key : (device -> 'a) -> 'a key
(** [new_key init] is {!Rig.Local.new_key}. *)

val local : 'a key -> device -> 'a
(** [local k d] is {!Rig.Local.get}. *)

val drop_values : device -> unit
(** [drop_values d] drops every key's value on [d]. A get on [d] from then
    on stores nothing. The call that runs [d]'s stop calls it once the
    driver's stop returned. *)

(** {1:loss Loss and stops}

    A counted call is a driver call on a live device that the device's stop
    never overlaps ({!counted}). Every driver call on a live device is
    counted except [locate] and [peer].

    A device's state is one word of its C line:
    {v
    Live ─loss─> Losing ─spread─> Owed ─claimed─> Stopping ─returned─> Ended
    any state, in a forked child, for a driver's device ─> Orphaned
    v}
    A device is lost once. The first loss, by {!lose}, {!close}, {!fail} or
    a failure of the driver's [submit] or [commit], records its reason and
    the value the word read at the loss. It then spreads to every device
    whose unreached work waits in its queue on an unreached value of the
    lost one, and from each of those in turn. A device lost by this spread
    has the reason ["<name> lost"], where <name> is the device it waited on,
    or, under {!fail}, [fail]'s reason. The stop is then owed. It runs once
    no counted call on the device is in flight, in a call that acts for the
    device ({!run_own}): its counted calls but a wait's, a submit on it, a
    commit or settle that fails, a counted call it refuses as lost, a drain
    of its memory, its {!close}, its publication, an open of its name and
    {!fail}. Every round of reclamation and an open that finds every slot
    held drain every device. A wait runs none: an interrupt there half-runs
    no driver's stop. An Ended device is {e Stopped} once its word reads its
    last submitted value. An io device is Stopped as soon as it is Ended,
    since its reads and writes run inside the calls that ask for them. Work
    a driver's device took before the loss may run until then, and only
    then are its memory and objects given back. Its word, records and line
    stay for the process.

    A forked child makes every lock anew. Each driver's device it inherited
    is Orphaned, and lost with the reason ["forked"] unless it was lost
    already. Its stop never runs, its word is never read, and its objects are
    forgotten without a call, since its work is the parent's. An io device
    keeps its state, and a stop that a thread of the parent had claimed is
    owed again. *)

val is_lost : device -> bool
(** [is_lost d] is [true] iff [d] is lost. It is [false] for the host
    without reading a record. *)

val orphaned : device -> bool
(** [orphaned d] is [true] iff [d] is a driver's device this process
    inherited from the parent it was forked from. *)

val lost : device -> string option
(** [lost d] is {!Rig.lost}[ d]. *)

val raise_lost : device -> 'a
(** [raise_lost d] raises {!Lost} with the reason of [d]'s loss. [d] must be
    lost. *)

val stop_returned : device -> bool
(** [stop_returned d] is [true] iff [d] is Ended or Orphaned. *)

val stopped : device -> bool
(** [stopped d] is [true] iff [d] is Stopped: Ended, and its word reads its
    last submitted value, so no work of [d] runs and its objects may be
    given back. For a word of its driver's own it asks the driver for the
    word under [d]'s lock, while [d] holds its slot, so its caller does not
    hold that lock. *)

val free_slot : device -> bool
(** [free_slot d] frees [d]'s slot if [d] is Stopped and nothing of it
    remains to give back: no token of its release list that its drains have
    not routed, nothing in its cache, among its retiring memory or its
    retired mappings, or waiting for a value. [d] must not be the process's first {!failure}, which
    keeps its slot. A [d] lost before it reached every value it was given
    leaves those values on the slot's line as a range that holds [d], so
    its dropped points raise its loss for good. It is whether this call
    freed the slot, [false] for a slot freed before. A later open takes the
    slot: [d] answers from then on as it stood, and its points from the
    slot's line. *)

val answered : (device -> unit) ref
(** [answered] is called with each device whose stop returned, by the call
    that ran the stop. {!Memory} sets it to {!Memory.drain}. *)

val run_own : device -> unit
(** [run_own d] runs [d]'s stop if it is owed and [d] has no counted call in
    flight, then {!answered}. It is one load while [d]'s stop is not owed. A
    fault of the stop is dropped. Another exception the stop raises is
    raised again once the state has moved on, without running {!answered}.
    An exception that lands after the stop is claimed and before the
    driver's stop is called owes the stop again. *)

val lose : device -> string -> 'a
(** [lose d why] loses [d] for the reason [why] if [d] is live, which makes
    the loss the process's {!failure} if none came before, and raises
    {!Lost} with the reason of [d]'s first loss. A wait calls it, so it runs
    no stop. *)

(** Which exceptions besides a fault lose a device when they escape a counted
    call: none, every one, or every one but a constructor's, which are the
    caller's. *)
type scope = Faults | Any | Any_but of Obj.Extension_constructor.t

val counted : device -> scope -> (unit -> 'a) -> 'a
(** [counted d scope f] is [f ()] as a counted call on [d], so [d]'s stop
    runs only once no counted call is in flight. A fault that [f] raises
    loses [d] and raises {!Lost}. Another exception loses [d] with its text
    if [scope] names it, and passes through either way. The loss and the
    end of the count are one C call, the handler's first step. Raises
    {!Lost} without calling [f] if [d] is lost, running [d]'s owed stop
    first. *)

(** The outcome of a {!give}: the object went back, or must go back after
    [d]'s stop. *)
type given = Given | Later

val give : device -> scope -> (unit -> unit) -> given
(** [give d scope f] gives an object back to [d]'s driver through [f]. It is
    [Later], without calling [f], if [d] is lost and not yet Stopped. On a
    live device it is a counted call under [scope], as for {!counted}, but
    a fault of [f] is not raised. On a Stopped device it is uncounted, and
    drops a fault or a [Sys_error] of [f]. An Orphaned device's objects are
    forgotten: [give] is [Given] without calling [f].

    [Later] runs no stop: the caller puts the object back where
    a drain finds it, then calls {!run_own}, so that the stop's drain meets
    the object in its place, ahead of what waits behind it. *)

val close : device -> unit
(** [close d] is {!Rig.close}[ d]. *)

val fail : string -> unit
(** [fail why] is {!Rig.fail}[ why]. *)

val failure : unit -> string option
(** [failure ()] is {!Rig.failure}[ ()]. *)

(** {1:timeline Timelines}

    A point of a device is done once the device's word has read its value
    while the device was live. For a lost device the value read at the loss
    counts, so a point whose value the word shows only after the loss is
    dropped and not done, and a wait for it raises {!Lost}. A point is
    settled once no work up to it can touch memory: the device's word reads
    its value now, after a loss too, or the device is Orphaned, whose work
    is the parent's. Waits and
    checks ask whether a point is done ({!is_done}); the return of memory
    asks whether it is settled ({!settled}). A lost device's points are
    answered from its line: {!is_done}, {!settled}, {!check} and
    {!wait_point} on them allocate nothing unless they raise. *)

val submitted : device -> int
(** [submitted d] is {!Rig.submitted}[ d]. *)

val next_value : device -> int
(** [next_value d] is one past [d]'s last submitted value, read under [d]'s
    C mutex, under which every submit checks its waits: every submit whose
    check came before has a value at most this one. A device whose slot
    passed on answers one past its final value. *)

val committed : device -> int
(** [committed d] is the last value of [d] known committed, by {!commit} or
    by a driver's [submit] that returned [RIG_COMMITTED]; [0] before any. *)

val word : device -> int
(** [word d] is the last value [d]'s word showed: read in its slot, or, for
    a word of its driver's own, asked of the driver as a {!counted} call,
    which may raise {!Lost}. It is [d]'s base on an io device. For a lost
    device whose word is its driver's own it is the last value read. *)

(** What [caml_rig_submit] and [caml_rig_commit] in [rig_device.c] return.
    The C enum also returns that the device is lost, before or after its
    driver's [submit], which {!Submission.submit} raises as {!Lost}. *)
module Answer : sig
  val ok : int
  (** The driver took the work, or committed it. *)

  val busy : int
  (** Another call is submitting or committing on the device; nothing was
      done. *)

  val no_room : int
  (** The device's queues have no room until a later value is reached. *)

  val never : int
  (** The parts never fit the device's queues. *)

  val producer_lost : int
  (** A device the work waits for in the queue is lost. *)

  val failed : int
  (** The driver's [submit] or [commit] failed, and this call lost the
      device. *)

  val no_memory : int
  (** The host has no memory to record the work's waits in the device's
      queues. *)

  val grow : int
  (** The driver's room answered that the parts fit once its queues grow
      ({!Rig_edge.Driver.grow}). *)
end

val commit : device -> int -> unit
(** [commit d v] commits [d]'s submitted work unless [v] is committed,
    waiting while another call is submitting or committing on [d]. A commit
    that fails loses [d] and runs its stop; on a lost device it does
    nothing. *)

val settle : device -> int -> bool
(** [settle d v] is [true] iff [d]'s driver wrote the times of [d]'s value
    [v], which its submit was given while a profile was taken. It has the
    driver write them, waiting while another call is submitting or
    committing on [d]. [v] must be reached. A settle that fails loses [d]
    and runs its stop; on a lost device it does nothing and is [false]. *)

val signaled : device -> int
(** [signaled d] is {!Rig.signaled}[ d]. *)

val wait : device -> int -> unit
(** [wait d v] commits [d]'s work if [v] is not committed and blocks until
    [d] reached [v]. Raises [Invalid_argument] if [v] exceeds [submitted d],
    and {!Lost} if [d] is or becomes lost. *)

val wait_value : device -> int -> unit
(** [wait_value d v] is {!wait} for a value [d] assigned, such as a point's,
    which it does not check against [submitted d]. *)

val wait_point : int -> bool -> int -> unit
(** [wait_point st every p] is {!wait} on [p]'s device and value, committing
    as it does, except that a point that is done returns, also on a device
    lost since, and a point of a device that gave its slot to a later one
    raises {!Lost} only if its loss dropped it. [st] is [0] or the stamps
    that [p] is a point of, which the caller keeps reachable, an access
    following their uses' points if [every], their writes' otherwise. It
    raises {!Lost} if [p] is not done and its device is or becomes lost, or,
    before it blocks, if a lost device did not reach a point of [st] that
    the access depends on. It never raises [Invalid_argument]. *)

val is_done : int -> bool
(** [is_done p] is [true] iff [p] is done. It never raises {!Lost}, since a
    read of the word that finds [p]'s device lost answers from its line. It
    raises [Invalid_argument] where the driver refuses the read, and another
    exception of the driver's that is no fault, as {!word} does. A point of
    a device that gave its slot to a later one is done. *)

val reached_point : int -> bool
(** [reached_point p] is {!Rig.Point.reached}: {!is_done}[ p], after a commit
    of [p]'s value as {!wait_point} commits it, which waits for no work. It
    raises {!Lost} if [p] is not done and its device is lost. *)

val settled : int -> bool
(** [settled p] is [true] iff [p] is settled. A point whose device the read
    of its word loses has not settled. A point of a device that gave its slot
    to a later one is settled: the slot's free required its device to have
    stopped. *)

val check : int -> unit
(** [check p] raises {!Lost} if [p]'s device is lost and [p] is not done. *)

val raise_point_lost : int -> 'a
(** [raise_point_lost p] raises {!Lost} for [p], which a lost device did not
    reach, with that device: the device of [p]'s slot, or the earlier one
    whose lost range of the slot holds [p]. *)

val release_list : unit -> int
(** [release_list ()] is a new, empty C release list, never freed. *)
