(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GSP's message queues (private).

    Two rings of 4 KiB elements in system memory: the command queue, which the
    process writes and the GSP reads, and the status queue back. A queue's
    header holds its writer's position; its reader keeps its own in the other
    queue's header page. A message is a record of one or more elements: the
    element's header (checksum, sequence number, element count), the RPC's
    header (signature, function, result, length), then its body. A body longer
    than 16 elements continues in records of the function [CONTINUATION_RECORD].
    A record's checksum makes the XOR of its 64-bit words folded to 32 bits zero
    ([message_queue_cpu.c]).

    The encoders are pure; {!send} and {!receive} access the rings, through
    windows on this machine or another, and never wait. *)

(** {1:codec Records} *)

val checksum : string -> int
(** [checksum s] is the XOR of [s]'s little-endian 64-bit words, [s] padded with
    zeros to a multiple of 8 bytes, folded to 32 bits. *)

val records : int -> string -> (int * string) list
(** [records fn body] is the records [body] is sent as: [fn] with the first
    bytes that fit 16 elements, then [CONTINUATION_RECORD]s with the rest. *)

val element : seq:int -> int -> string -> string
(** [element ~seq fn body] is the record of function [fn] with [body] and
    sequence number [seq], whole elements long, with its checksum. *)

type message = { fn : int; result : int; body : string }
(** The type for messages the GSP sends: its function or event, its result ([0]
    for success) and its body. *)

val message : string -> (message * int, string) result
(** [message s] is the message that starts the bytes [s] of a ring and the
    number of elements it takes, or [Error] if its header is not a GSP's. *)

val fault : message -> string option
(** [fault m] is the report of a fault of the GPU's work that [m] carries,
    naming the channel and the error: a channel the GSP stopped ([RC_TRIGGERED],
    its exception type named as [nverror.h] names it), or a fault the MMU queued
    ([MMU_FAULT_QUEUED]). It is [None] for any other message. An error log
    ([OS_ERROR_LOG]) is [None]: the GSP stops a channel itself and says so with
    [RC_TRIGGERED] ([_kgspRpcRCTriggered]), and logs errors that stop nothing,
    such as a retired page. *)

(** {1:queues Queues} *)

type t
(** The type for the two queues of one GSP. *)

val create : Device_pci.Window.t -> doorbell:Device_pci.Window.t -> t
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

val receive : t -> message option
(** [receive q] is the next message of the status queue, if the GSP wrote one,
    which it consumes.

    Raises [Invalid_argument] if [q] is not {!ready}. *)
