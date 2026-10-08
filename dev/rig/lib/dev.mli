(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices: the table, opening, counted calls, loss and stops, and waits. Any
   domain may call any function. *)

open Def

exception Lost of device * string
exception Out_of_memory of device * int

val host : device
val protect : device -> (unit -> 'a) -> 'a
(* [protect d f] is [f ()] run holding [d]'s lock, which guards [d]'s mutable
   fields. A forked child makes every lock anew. *)

val hold : device -> unit
val release : device -> unit
(* [hold d] holds [d]'s lock and [release d] gives it back, around a section
   that raises nothing. *)

val busy : device -> bool
(* [busy d] is [true] if a call holds [d]'s lock. *)

val of_index : int -> device
val iter : (device -> unit) -> unit
(* [iter f] is [f] over the open devices other than the host, by index. *)

val is_host : device -> bool
val is_io : device -> bool
val is_lost : device -> bool
val lost : device -> string option
val raise_lost : device -> 'a
val submitted : device -> int
val stop_returned : device -> bool
(* [stop_returned d] is [true] iff the lost [d]'s stop returned. *)

val stopped : device -> bool
(* [stopped d] is [true] iff [d] is lost and counts as stopped: its stop
   returned and its word reads its last submitted value, so no work of [d] runs
   and its memory may be freed. *)

val inherited : device -> bool
(* [inherited d] is [true] iff [d] is a driver's device this process inherited
   from the parent it was forked from. *)

val same_machine : device -> device -> bool
val host_of : device -> device
val reaches : device -> device -> bool
val answered : (device -> unit) ref
(* [answered] runs once a lost device's stop answered: the memory module sets it
   to free what the answer makes due. *)

val stop_claimed : int array -> unit
(* [stop_claimed a] runs the stops of the devices whose indices [a] holds from
   its second element on, as the C loss and submit answer them. *)

val stop : device -> unit
(* [stop d] runs the claimed stop of [d] and records its answer. *)

val lose : device -> string -> 'a
val counted : device -> (unit -> 'a) -> 'a
val word : device -> int
val wait : device -> int -> unit
val after : device -> int -> (unit -> unit) -> unit
val point_reached : int -> bool
val release_list : unit -> int

val open_driver :
  ?memory_device:bool ->
  (module Sigs.Driver with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result

val open_io :
  (module Sigs.Io with type t = 'a) ->
  ?machine:string ->
  ?host:bool ->
  name:string ->
  (unit -> ('a, string) result) ->
  (device, string) result
