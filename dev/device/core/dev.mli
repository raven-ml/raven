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
val of_index : int -> device
val all : unit -> device array

(* [all ()] is the devices by index; an index whose open failed holds the
   host. *)
val is_host : device -> bool
val is_io : device -> bool
val is_lost : device -> bool
val lost : device -> string option
val raise_lost : device -> 'a
val submitted : device -> int
val answer : device -> int
(* [answer d] is [d]'s stop's answer: 0 none, then [answer_stopping],
   [answer_stopped] or [answer_unknown]. *)

val answer_stopping : int
val answer_stopped : int
val answer_unknown : int
val forked : unit -> bool
val upgrade : device -> bool
(* [upgrade d] records [Stopped] in place of [Unknown] once [d]'s word reads its
   submitted value, and is [true] if this call recorded it. *)

val copies : device -> bool
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
val signaled : device -> int
val wait : device -> int -> unit
val after : device -> int -> (unit -> unit) -> unit
val point_reached : int -> bool
val still_ms : int
val release_list : unit -> int
val now_ms : unit -> int

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
