(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices: the table of open devices, opening, counted calls, loss and stops,
    and waits on a timeline.

    Any domain may call any function. A device's lock ({!protect}) guards its
    mutable fields; no driver call and no wait runs under it, and it is not
    reentrant.

    A device is lost once: the first loss, by {!lose} or a failed hand-over,
    records why and spreads to the devices whose queues wait on its unreached
    work. A lost device's stop runs once no {!counted} call on it is in flight,
    claimed by the call that lost it, by the counted call that leaves last, or
    by the next counted call or submit on it.

    A forked child makes every lock anew. Each driver's device it inherited is
    lost, with the reason ["forked"]: its stop counts as returned and never
    runs, and the device never counts as {!stopped}. Its io devices stay open.

    Functions that read a device's C record ({!raise_lost}, {!lose}, {!stop})
    take a device other than the host. *)

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

val lost : device -> string option
(** {!Rig.lost}. *)

val raise_lost : device -> 'a
(** [raise_lost d] raises {!Lost} with the reason of [d]'s loss. [d] is lost. *)

val submitted : device -> int
(** {!Rig.submitted}. *)

val stop_returned : device -> bool
(** [stop_returned d] is [true] iff [d] is lost and its stop returned, or [d] is
    a driver's device a forked child inherited. *)

val stopped : device -> bool
(** [stopped d] is [true] iff [d] is lost and counts as stopped: its stop
    returned and, for a driver's device, its word reads its last submitted
    value, so no work of [d] runs and its memory may be freed. Behind a
    transport it asks the driver for the word. *)

val inherited : device -> bool
(** [inherited d] is [true] iff [d] is a driver's device this process inherited
    from the parent it was forked from. *)

val same_machine : device -> device -> bool

val host_of : device -> device
(** {!Rig.host_of}. *)

val reaches : device -> device -> bool
(** {!Rig.reaches}. *)

val answered : (device -> unit) ref
(** [answered] runs once a lost device's stop answered, from {!stop}: {!Memory}
    sets it to free what the answer makes due. *)

val stop_claimed : int array -> unit
(** [stop_claimed a] runs the {!stop} of the devices whose indices [a] holds
    from its second element on, as the C loss and submit answer them. *)

val stop : device -> unit
(** [stop d] runs [d]'s driver's stop, which the caller claimed, records its
    answer, then runs {!answered}. A fault of the stop is dropped; another
    exception it raises is raised again once the answer is recorded, without
    running {!answered}. *)

val lose : device -> string -> 'a
(** [lose d why] loses [d] with [why] unless it is lost, runs the stops the loss
    claimed and raises {!Lost} with the first loss's reason. *)

val counted : device -> (unit -> 'a) -> 'a
(** [counted d f] is [f ()] as a counted call on [d]: [d]'s stop runs only once
    no counted call is in flight. A driver fault that [f] raises loses [d].
    Raises {!Lost} without calling [f] if [d] is lost, running its stop first if
    it is due. On the host it is [f ()]. *)

val word : device -> int
(** [word d] is the last value [d]'s word showed: read at its host address, or,
    behind a transport, asked of the driver as a {!counted} call, which may
    raise {!Lost}; [0] on the host and an io device. A lost device behind a
    transport answers the last value read. *)

val wait : device -> int -> unit
(** [wait d v] blocks until [d] reached [v], then runs the {!after} functions
    due, raising the first exception one raised once all ran. Raises
    [Invalid_argument] if [v] exceeds [submitted d], and {!Lost} if [d] is or
    becomes lost. *)

val after : device -> int -> (unit -> unit) -> unit
(** [after d v f] runs [f] in the first {!wait} on [d] that finds [v] reached,
    on that wait's domain. Nothing else runs it. *)

val point_reached : int -> bool
(** [point_reached p] is [true] iff [p]'s device's word reads [p]'s value, and
    for a point on the host. [false] in a forked child for a device it
    inherited. It may raise {!Lost} as {!word} does. *)

val release_list : unit -> int
(** [release_list ()] is a new, empty C release list, never freed. *)

val open_driver :
  ?memory_device:bool ->
  (module Sigs.Driver with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
(** [open_driver (module D) ~machine ~name make] is {!Rig.open_}. Concurrent
    opens of a name wait for the first. A fault while the device's facts are
    read stops the handle and is [Error]. With [memory_device] (defaults to
    [false]), {!Rig.runs_on_host} is [true] of the device and misuse raises
    under {!Rig.memory_device}'s name. *)

val open_io :
  (module Sigs.Io with type t = 'a) ->
  ?machine:string ->
  ?host:bool ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
(** {!Rig.open_io}. A fault while the device's budget is read stops the handle
    and is [Error]. *)
