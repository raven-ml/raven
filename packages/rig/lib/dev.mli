(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dev keeps the table of open devices, and runs each device's timeline
    and loss.

    A device is four records. The C line ([struct rig_line] in
    [rig_stubs.h]) holds its timeline, its state and its loss, and lasts for
    the life of the process, since a point names its device by index. The
    OCaml record ({!Def.device}) holds its facts: the table holds it until
    its resources are freed, and weakly from then on, so a point finds it
    while a value of the program refers to it, and {!forgotten} once the
    collector took it, when the line's reason goes. Two are its resources,
    freed at its end ({!free_live}): the OCaml {!Def.live} holds its memory
    lists and what only its work needs, and the C record ([struct
    rig_device]) its counted calls, its turn and the waits its queues hold
    on other devices. This module implements {!Rig}'s opening and closing
    ({!Rig.section-devices}), waits on a timeline ({!Rig.section-timeline})
    and loss ({!Rig.section-loss}).

    Any domain may call any function. A device's lock ({!protect}) guards the
    mutable fields of its {!Def.live}. No driver call and no wait runs under
    it, and it is not reentrant. The host is never lost. The life of a
    device from its loss to its stop, and what a fork does, are in
    {!section-loss}. {!Rig.section-timeline} and {!Rig.section-loss} state
    what timelines and a loss mean to a caller. *)

open Def

exception Lost of device * string
(** {!Rig.Lost}. *)

exception Out_of_memory of device * int
(** {!Rig.Out_of_memory}. *)

val host : device
(** [host] is {!Rig.host}, of index [0]. *)

(** {1:locks Locks}

    Each raises {!Lost} if [d]'s resources are freed. *)

val protect : device -> ('a -> 'b) -> 'a -> 'b
(** [protect d f x] is {!Lock.protect} of [d]'s lock. *)

val busy : device -> bool
(** [busy d] is [true] if a call held [d]'s lock when [busy] tried to take
    it. *)

(** {1:table The table of devices} *)

val forgotten : device
(** [forgotten] is {!Rig.Point.device}'s stand-in for a device the collector
    took: lost with the reason ["ended"], Stopped, its word and submitted
    value [max_int], its budget [0] until set. Its index is past every
    point's. *)

val of_index : int -> device
(** [of_index i] is the device of index [i], {!host} for [0], or {!forgotten}
    once the collector took that device. [i] must be the index of a device
    made. It allocates once that device's resources are freed. *)

val iter : (device -> unit) -> unit
(** [iter f] applies [f] to each published device other than the host whose
    resources are not freed ({!freed}), lost ones included, in index order.
    It costs the devices it visits, however many the process opened. *)

val live : device -> live
(** [live d] is [d]'s resources. Raises {!Lost} if they are freed. *)

val freed : device -> bool
(** [freed d] is [true] iff [d]'s resources are freed. *)

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
  (unit -> ('a, string) result) ->
  (device, string) result
(** [open_driver (module D) ~machine ~name make] is {!Rig.open_} on
    [machine], or {!Rig.open_host} for [Hosts]. Concurrent opens of one name
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
(** [drop_values d] drops every key's value on [d], and every value stored
    for [d] from then on. {!stop} calls it once the driver's stop returned. *)

(** {1:loss Loss and stops}

    A counted call is a driver call on a live device that the device's stop
    never overlaps ({!counted}). Every driver call on a live device is
    counted except [locate] and [peer].

    A device's state is one word of its C line:
    {v
    Live ─loss─> Losing ─spread─> Owed ─claimed─> Stopping ─returned─> Ended
    Ended ─the word reads the last submitted value─> Stopped
    Stopping ─returned, for an io device─> Stopped
    any state, in a forked child, for a driver's device ─> Orphaned
    v}
    A device is lost once. The first loss, by {!lose}, {!close}, {!fail} or
    a failure of the driver's [submit] or [commit], records its reason and
    the value the word read at the loss. It then spreads to every device
    whose unreached work waits in its queue on an unreached value of the
    lost one, and from each of those in turn. A device lost by this spread
    has the reason ["<name> lost"], where <name> is the device it waited on,
    or, under {!fail}, [fail]'s reason. The stop is then owed, and
    {!run_owed} runs it once no counted call on the device is in flight. An
    io device is Stopped once its stop returns. A driver's device stays
    Ended until its word reads its last submitted value, since work it took
    before the loss may still run, and only then are its memory and objects
    given back.

    A Stopped device's timeline word ends in three steps, each taken by a
    drain ({!Memory.drain}):
    + While another device's queue holds a wait on the word that its own
      word has not passed, the word stays, since that queue still reads it.
    + Then {!move_word} points the C line's readers at a copy of the
      word's last value in the line. Every later read reads the copy,
      including the read for a point a stamp names, so the word can end
      while stamps still name its values.
    + The driver gets the word back once a minor collection has run since
      the move, after every other device's mapping of it. The collection
      waits for every domain that holds its runtime lock, so a reader that
      loaded the word's address before the move has finished with it. A
      reader without the runtime lock reads under the device's mutex, or,
      as a spin on the word, counts as a call in flight on the device
      ([inside] in the C record), which the stop and the move wait for.

    Once its word is given back, its memory has all returned and no drain
    is in flight on it, a Stopped device's resources are freed
    ({!free_live}). The C record goes once neither its custom block nor a
    prepared submission holds it, and its lock once collected, so a call
    that read them before the free finishes with them. From then on the
    device answers from its facts and its line, and a use of its resources
    raises {!Lost}.

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
(** [stop_returned d] is [true] iff [d] is Ended, Stopped or Orphaned. *)

val stopped : device -> bool
(** [stopped d] is [true] iff [d] is Stopped: its stop returned and, for a
    driver's device, its word reads its last submitted value, so no work of
    [d] runs and its objects may be given back. For a word with no host
    address it asks the driver for the word. *)

val answered : (device -> unit) ref
(** [answered] is called with each device whose stop returned, from
    {!run_owed}. {!Memory} sets it to {!Memory.drain}. *)

val next_ended : int -> int
(** [next_ended i] is the least index from [i] on of a device that is Ended
    or Stopped and whose resources are not freed, or [0] for none. Such a
    device's memory and mappings may go back long after its stop. A forked
    child's orphaned devices ({!orphaned}) are never among them: nothing
    drains them. The stop's change of state makes a device one of these, so
    nothing an exception interrupts can leave a stopped device out. While
    there are none it reads one count, and it allocates nothing. *)

val free_live : device -> unit
(** [free_live d] frees [d]'s resources: {!live} raises {!Lost} from then
    on, and [d] leaves {!ended}, its name's slot in the table and the
    values it dropped ({!drop_values}), and the table holds it weakly. [d]
    is Stopped, its word is given back, its memory has all returned and no
    drain is in flight on it; {!Memory} decides when. *)

val run_owed : unit -> unit
(** [run_owed ()] runs each owed stop whose device has no counted call in
    flight, then {!answered}. A loss, the end of a counted call, a submit,
    and a counted call that finds its device lost each call it. It is one
    load while no stop is owed. A fault of a stop is dropped. Another
    exception a stop raises is raised again once the state has moved on,
    without running {!answered}. *)

val lose : device -> string -> 'a
(** [lose d why] loses [d] for the reason [why] if [d] is live, which makes
    the loss the process's {!failure} if none came before. It then runs the
    owed stops and raises {!Lost} with the reason of [d]'s first loss. *)

val counted : device -> (unit -> 'a) -> 'a
(** [counted d f] is [f ()] as a counted call on [d], so [d]'s stop runs
    only once no counted call is in flight. A driver fault that [f] raises
    loses [d] and raises {!Lost}; another exception passes through. Raises
    {!Lost} without calling [f] if [d] is lost, running the owed stops
    first. *)

val give : device -> (unit -> unit) -> bool
(** [give d f] gives an object back to [d]'s driver through [f], and is
    [false], without calling [f], if [d] is lost and not yet Stopped; the
    caller must give the object again after the stop. On a live device it
    is a counted call, and a fault of [f] loses [d] without being raised. On
    a Stopped device it is uncounted, and drops a fault or a [Sys_error] of
    [f]. An Orphaned device's objects are forgotten: [give] is [true]
    without calling [f]. *)

val move_word : device -> bool
(** [move_word d] moves the readers of the Stopped [d]'s word to a copy in
    its C record, once no record of another device holds a wait on the word
    that its own word has not passed. A reader that holds a domain's
    runtime lock may still read the driver's word until that domain has
    passed a minor collection ({!minors}). It is [true] once, for the call
    that moved them. *)

val minors : unit -> int
(** [minors ()] is the number of minor collections the program made. Each
    waits for every domain that holds its runtime lock. *)

val close : device -> unit
(** [close d] is {!Rig.close}[ d]. *)

val fail : string -> unit
(** [fail why] is {!Rig.fail}[ why]. *)

val failure : unit -> string option
(** [failure ()] is {!Rig.failure}[ ()]. *)

val generation : unit -> int
(** [generation ()] is the number of forks that made this process from the
    first. A value made in another generation was made before a fork. *)

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
    answered from its line, without {!of_index}: {!is_done}, {!settled},
    {!check} and {!wait_point} on them allocate nothing unless they raise. *)

val submitted : device -> int
(** [submitted d] is {!Rig.submitted}[ d]. *)

val committed : device -> int
(** [committed d] is the last value of [d] known committed, by {!commit} or
    by a driver's [submit] that returned [RIG_COMMITTED]; [0] before any. *)

val word : device -> int
(** [word d] is the last value [d]'s word showed: read at its host address,
    or, for a word with no host address, asked of the driver as a {!counted}
    call, which may raise {!Lost}. It is [0] on the host and an io device.
    For a lost device whose word has no host address it is the last value
    read. *)

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

  val need_record : int
  (** The device's record of waits in its queues must grow first. *)
end

val commit : device -> int -> unit
(** [commit d v] commits [d]'s submitted work unless [v] is committed,
    waiting while another call is submitting or committing on [d]. A commit
    that fails loses [d]; on a lost device it does nothing. *)

val settle : device -> int -> unit
(** [settle d v] has [d]'s driver write the times of [d]'s value [v], which
    its submit was given while a profile was taken, waiting while another
    call is submitting or committing on [d]. [v] must be reached. On a lost
    device it does nothing. *)

val signaled : device -> int
(** [signaled d] is {!Rig.signaled}[ d]. *)

val wait : device -> int -> unit
(** [wait d v] commits [d]'s work if [v] is not committed, blocks until [d]
    reached [v], then runs the {!after} functions due, raising the first
    exception one raised once all have run. Raises [Invalid_argument] if [v]
    exceeds [submitted d], and {!Lost} if [d] is or becomes lost. *)

val wait_value : device -> int -> unit
(** [wait_value d v] is {!wait} for a value [d] assigned, such as a point's,
    which it does not check against [submitted d]. *)

val wait_point : memory -> bool -> int -> unit
(** [wait_point m every p] is {!wait} on [p]'s device and value, committing
    as it does, except that a point that is done returns, also on a device
    lost since. [p] is a point of [m]'s stamps that an access follows: their
    last write's and, if [every], their uses'. It raises {!Lost} if [p] is
    not done and its device is or becomes lost, or, before it blocks, if a
    lost device did not reach a point of [m]'s stamps that the access
    depends on. It never raises [Invalid_argument]. *)

val after : device -> int -> (unit -> unit) -> unit
(** [after d v f] runs [f] in the first {!wait} or {!wait_point} on [d] that
    finds [v] reached, on that wait's domain. Nothing else runs it. *)

val is_done : int -> bool
(** [is_done p] is [true] iff [p] is done. It may raise {!Lost} as {!word}
    does. *)

val settled : int -> bool
(** [settled p] is [true] iff [p] is settled. A point whose device the read
    of its word loses has not settled. *)

val check : int -> unit
(** [check p] raises {!Lost} if [p]'s device is lost and [p] is not done. *)

val release_list : unit -> int
(** [release_list ()] is a new, empty C release list, never freed. *)
