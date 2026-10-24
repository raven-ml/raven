(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP's two message queues in system memory: the command queue, which
    the process writes and the GSP reads, and the status queue back. {!send},
    {!next} and {!consume} reach them through windows on this machine or
    another, and never wait. *)

type t
(** The type for the two queues of one GSP. *)

val create : Rig_pci.Window.t -> doorbell:Rig_pci.Window.t -> t
(** [create w ~doorbell] lays out the two queues in [w], the command queue
    first, and writes the command queue's header, before the GSP boots.
    [doorbell] is the GSP's queue head register, which {!send} writes. *)

val ready : t -> bool
(** [ready q] is [true] once the GSP has written the status queue's header,
    which it does once it runs. *)

val send : ?call:bool -> t -> int -> string -> bool
(** [send ~call q fn body] writes the records of [fn] with [body] and writes
    the doorbell after each, or is [false], writing nothing, if the command
    queue has no room for them. If [call] (defaults to [false]), [fn] is
    {!owed} from the write that publishes the last record. *)

val next : t -> (Rpc.message, string) result option
(** [next q] is the next message of the status queue, if the GSP wrote one:
    [Error] if its elements are not a GSP message ({!Rpc.message}). It stays
    in the queue until {!consume}, so a caller handles a message before it
    consumes it.

    Raises [Invalid_argument] if [q] is not {!ready}. *)

val consume : t -> unit
(** [consume q] gives the GSP back the elements of the message the last
    {!next} read: the whole message, or one element when its count is out of
    range. It does nothing if [next] read none since the last [consume]. A
    message of the {!owed} function ends it, from the write that gives it
    back.

    Raises [Invalid_argument] if [q] is not {!ready}. *)

val owed : t -> int
(** [owed q] is the function of the last [~call:true] {!send} until a
    {!consume} gives back a readable message of that function, or [-1]. With
    one call outstanding at a time, as the GSP's caller keeps, and the GSP
    answering calls in order, the first message of that function is its
    answer. An answer {!next} cannot read leaves the function owed. *)
