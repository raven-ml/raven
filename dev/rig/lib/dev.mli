(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices: the table of open devices, opening, counted calls, loss and stops,
    and waits on a timeline.

    Any domain may call any function. A device's lock ({!protect}) guards its
    mutable fields; no driver call and no wait runs under it, and it is not
    reentrant.

    A device's state is one word of its C record:

    {v
      Live ──loss──> Stopping ──stop returned──> Ended ──last value──> Stopped
        │                                                  in the word
        └──fork, in the child, for a driver's device──> Orphaned
    v}

    A device is lost once: the first loss, by {!lose}, {!close}, {!fail} or a
    failed hand-over, records why and spreads to the devices whose queues wait
    on its unreached work. Its stop is then owed, and runs once no {!counted}
    call on it is in flight, from {!run_owed}: after the loss, after a counted
    call or a submit, and when a counted call finds the device lost. An io
    device is Stopped once its stop returned.

    A forked child makes every lock anew. Each driver's device it inherited is
    Orphaned, lost with the reason ["forked"] unless it was lost: its stop never
    runs, its word is never read, and its objects are forgotten without a call.
    Its io devices stay as they were.

    A Stopped device's timeline word ends in three steps, each in a drain
    ([Memory]):
    + While another device's queue holds a wait on the word that its own word
      has not passed, the word stays: that queue still reads it.
    + Then {!move_word} points the C record's readers at a copy of the word's
      last value in the record. Every later read, a point of the device that a
      stamp names included, reads the copy, so stamps need not be gone.
    + The driver gets the word back ({!Rig.Driver.free}) once a minor collection
      ran since the move, after every other device's mapping of it. The
      collection is the grace period: it waits for every domain that holds its
      runtime lock, so a reader that loaded the word's address before the move
      has finished with it. A reader without the runtime lock reads under the
      device's mutex, or, as a spin on the word, counts as a call inside the
      device, which the stop and the move wait for.

    The host is never lost. *)

open Def

exception Lost of device * string
(** {!Rig.Lost}. *)

exception Out_of_memory of device * int
(** {!Rig.Out_of_memory}. *)

val host : device

val protect : device -> (unit -> 'a) -> 'a
(** [protect d f] is [f ()] run holding [d]'s lock. *)

val hold : device -> unit
(** [hold d] holds [d]'s lock, for a section that raises nothing, which
    {!release} ends. *)

val release : device -> unit

val busy : device -> bool
(** [busy d] is [true] if a call holds [d]'s lock. *)

val of_index : int -> device
(** [of_index i] is the device of index [i]: {!host} for [0] and for an index
    whose open failed. [i] is the index of a device made. *)

val iter : (device -> unit) -> unit
(** [iter f] is [f] over the devices made other than the host, lost ones
    included, by index. *)

val is_host : device -> bool
val is_io : device -> bool
val is_lost : device -> bool

val orphaned : device -> bool
(** [orphaned d] is [true] iff [d] is a driver's device this process inherited
    from the parent it was forked from. *)

val lost : device -> string option
(** {!Rig.lost}. *)

val raise_lost : device -> 'a
(** [raise_lost d] raises {!Lost} with the reason of [d]'s loss. [d] is lost. *)

val submitted : device -> int
(** {!Rig.submitted}. *)

val committed : device -> int
(** [committed d] is the last value of [d] known committed: by {!commit}, or
    by a hand-over that answered [RIG_COMMITTED]; [0] before any. *)

val stop_returned : device -> bool
(** [stop_returned d] is [true] iff [d] is Ended, Stopped or Orphaned. *)

val stopped : device -> bool
(** [stopped d] is [true] iff [d] is Stopped: its stop returned and, for a
    driver's device, its word reads its last submitted value, so no work of [d]
    runs and its objects may be given back. Behind a transport it asks the
    driver for the word. *)

val same_machine : device -> device -> bool

val host_of : device -> device
(** {!Rig.host_of}. *)

val reaches : device -> device -> bool
(** {!Rig.reaches}. *)

val answered : (device -> unit) ref
(** [answered] runs once a lost device's stop returned, from {!run_owed}:
    {!Memory} sets it to drain. *)

val ended : unit -> device list
(** [ended ()] is the devices whose stop returned, newest first: their memory
    and mappings may be given back long after the stop. *)

val run_owed : unit -> unit
(** [run_owed ()] runs each owed stop whose device has no counted call in
    flight, then {!answered}. It is one load while no stop is owed. A fault of a
    stop is dropped; another exception it raises is raised again once the state
    moved on, without running {!answered}. *)

val lose : device -> string -> 'a
(** [lose d why] loses [d] with [why] unless it is lost, which is the process's
    {!failure} if none came before, runs the owed stops and raises {!Lost} with
    the first loss's reason. *)

val counted : device -> (unit -> 'a) -> 'a
(** [counted d f] is [f ()] as a counted call on [d]: [d]'s stop runs only once
    no counted call is in flight. A driver fault that [f] raises loses [d].
    Raises {!Lost} without calling [f] if [d] is lost, running the owed stops
    first. *)

val give : device -> (unit -> unit) -> bool
(** [give d f] gives an object back to [d]'s driver through [f]: as a counted
    call on a live device, where a fault of [f] loses [d] and is not raised;
    uncounted on a Stopped one, dropping a fault and a [Sys_error] it raises;
    and not at all on an Orphaned one. It is [false], without calling [f], on a
    device lost and not Stopped, whose objects go back only after its stop:
    the caller gives the object again then. *)

val move_word : device -> bool
(** [move_word d] moves the readers of the stopped [d]'s word to a copy in its C
    record once nothing else reads the driver's word: no record of another
    device holds a wait on it that its word has not passed. Every reader holding
    a domain's runtime lock may still read the driver's word until that domain
    passed a minor collection ({!minors}). It is [true] once, for the call that
    moved them. *)

val minors : unit -> int
(** [minors ()] is the number of minor collections the program made. Each waits
    for every domain that holds its runtime lock. *)

val close : device -> unit
(** {!Rig.close}. *)

val fail : string -> unit
(** {!Rig.fail}. *)

val failure : unit -> string option
(** {!Rig.failure}. *)

val generation : unit -> int
(** [generation ()] is the number of forks that made this process from the
    first: a value made in another generation was made before a fork. *)

val word : device -> int
(** [word d] is the last value [d]'s word showed: read at its host address, or,
    behind a transport, asked of the driver as a {!counted} call, which may
    raise {!Lost}; [0] on the host and an io device. A lost device behind a
    transport answers the last value read. *)

(** What [rig_device.c]'s submit and commit answer. *)
module Answer : sig
  val ok : int
  val busy : int
  val no_room : int
  val never : int
  val producer_lost : int
  val failed : int
  val need_record : int
end

val commit : device -> int -> unit
(** [commit d v] commits [d]'s submitted work under [d]'s turn
    ({!Rig.Driver.edge}) unless [v] is committed, waiting for the turn.
    A commit that fails loses [d]; on a lost device it does nothing. *)

val signaled : device -> int
(** {!Rig.signaled}. *)

val wait : device -> int -> unit
(** [wait d v] commits [d]'s work if [v] is not committed, blocks until [d]
    reached [v], then runs the {!after} functions due, raising the first
    exception one raised once all ran. Raises [Invalid_argument] if [v] exceeds
    [submitted d], and {!Lost} if [d] is or becomes lost. *)

val wait_point : int -> unit
(** [wait_point p] is {!wait} on [p]'s device and value, committing as it does,
    except that a point that is done returns, also on a device lost since: it
    raises {!Lost} only if [p] is not done and its device is or becomes lost. *)

val after : device -> int -> (unit -> unit) -> unit
(** [after d v f] runs [f] in the first {!wait} on [d] that finds [v] reached,
    on that wait's domain. Nothing else runs it. *)

val is_done : int -> bool
(** [is_done p] is [true] iff the work up to [p] is done: [p]'s device's word
    reads [p]'s value, read before the loss for a lost device, and the device is
    not {!orphaned}. It may raise {!Lost} as {!word} does. *)

val settled : int -> bool
(** [settled p] is [true] iff no work up to [p] touches memory any more: it is
    done, or its device is {!orphaned}, whose work is the parent's. A device
    lost by the read of its word has not settled. *)

val check : int -> unit
(** [check p] raises {!Lost} if [p]'s device is lost and [p] is not done. *)

val release_list : unit -> int
(** [release_list ()] is a new, empty C release list, never freed. *)

val open_driver :
  ?memory_device:bool ->
  (module Rig_edge.Driver with type t = 'a) ->
  ?machine:string ->
  ?host:bool ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
(** [open_driver (module D) ~machine ~host ~name make] is {!Rig.open_}.
    Concurrent opens of a name wait for the first. A fault while the device's
    facts are read stops the handle and is [Error]. With [memory_device]
    (defaults to [false]), {!Rig.runs_on_host} is [true] of the device and
    misuse raises under {!Rig.memory_device}'s name. *)

val open_io :
  (module Rig_edge.Io with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
(** {!Rig.open_io}. A fault while the device's budget is read stops the handle
    and is [Error]. *)
