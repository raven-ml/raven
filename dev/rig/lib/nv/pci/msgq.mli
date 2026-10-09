(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP's message queues.

    Two rings of 4 KiB elements in system memory: the command queue, which the
    process writes and the GSP reads, and the status queue back. A queue's
    header holds its writer's position; its reader keeps its own in the other
    queue's header page. The rings hold records ({!Rpc.records}).

    {!send} and {!receive} access the rings, through windows on this machine or
    another, and never wait. *)

type t
(** The type for the two queues of one GSP. *)

val create : Rig_pci.Window.t -> doorbell:Rig_pci.Window.t -> t
(** [create w ~doorbell] lays out the two queues in [w], the command queue
    first, and writes the command queue's header, before the GSP boots.
    [doorbell] is the GSP's queue head register, which {!send} writes. *)

val ready : t -> bool
(** [ready q] is [true] once the GSP wrote the status queue's header, which it
    does once it runs. *)

val send : t -> int -> string -> bool
(** [send q fn body] writes the records of [fn] with [body] and writes the
    doorbell after each, or is [false], writing nothing, if the command queue
    has no room for them. *)

val receive : t -> (Rpc.message, string) result option
(** [receive q] is the next message of the status queue, if the GSP wrote one,
    which it consumes: [Error] if its elements are not a GSP message
    ({!Rpc.message}).

    Raises [Invalid_argument] if [q] is not {!ready}. *)
