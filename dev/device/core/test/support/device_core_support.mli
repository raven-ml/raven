(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Test drivers and probes for the core's suites. *)

(** A driver over host memory whose queue runs only when its {!run} or its sleep
    runs it: a wait that returns before it slept leaves work unrun. Every driver
    call is logged. *)
module Polled : sig
  include Device_core.Driver

  val make :
    ?capacity:int ->
    ?may_block:bool ->
    ?waits_host:bool ->
    ?answer:[ `Stopped | `Unknown ] ->
    unit ->
    t
  (** [make ()] is a device whose queue holds [capacity] parts (defaults to
      1024): beyond it [room] answers [`Later], or, with [may_block], submit
      waits for room. With [waits_host] its queue waits for host-written words.
      Its stop answers [answer] (defaults to [`Stopped]). *)

  val open_ :
    ?capacity:int ->
    ?may_block:bool ->
    ?waits_host:bool ->
    ?answer:[ `Stopped | `Unknown ] ->
    string ->
    Device_core.t * t
  (** [open_ name] opens a fresh device named [name]. *)

  val run : t -> int
  (** [run d] runs the queued submissions whose waits hold: how many ran. *)

  val queued : t -> int
  val submits : t -> int

  val fail : t -> unit
  (** [fail d] makes [d]'s next submit fail. *)

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s next sleep raise [Fault why]. *)

  val set_word : t -> int -> unit
  (** [set_word d v] writes [v] into [d]'s word. *)

  val log : t -> string list
  (** [log d] is [d]'s driver calls, oldest first: ["alloc"], ["free"],
      ["map_host"], ["unmap"], ["sleep"], ["stop"], ["image"], ["unload"]. *)
end

val bump : nativeint
(** [bump] is a fill adding 1 to the 64-bit word its argument points at. *)

val load : int -> int
(** [load a] reads the 64-bit word at the host address [a]. *)

val store : int -> int -> unit
