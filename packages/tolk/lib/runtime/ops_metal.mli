(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Metal's command queues, as batches encode them.

    A Metal device runs the programs of a batch from an
    {e indirect command buffer}: a list of dispatches that the device records
    once and runs as often as it is asked to. Its placeholder, a volatile
    placeholder of the device tagged [("mtl_icb", commands, header)] ({!icb}),
    holds each program's arguments, 256-byte aligned, then, from byte [header],
    the words the engine writes when it links the batch: the indirect command
    buffer, each of its commands, and each pipeline they use, one word each.

    The batch's host program encodes the indirect command buffer into command
    buffers of the device's command queue and commits them. It sends each
    Objective-C message through [objc_msgSend], a C function of the library
    ["metal"] ({!Hcq2.ccall}), with the receivers and selectors it loads from a
    placeholder of the device tagged ["mtl_sel"]: the words {!handles}, then
    {!selectors}, which the engine writes when it links the batch. The command
    buffer and its encoder pass between messages through volatile placeholders
    of the device tagged ["mtl_cb"] and ["mtl_enc"]. *)

(** {1:queues Command queues} *)

val queues : host:string -> arch:string -> residency_set:bool -> Hcq2.queues
(** [queues ~host ~arch ~residency_set] is the command queues of a Metal device
    whose GPU family is [arch], such as ["Apple9"] or ["Mac2"], with a residency
    set of its buffers iff [residency_set], and whose batches are submitted by
    host programs of [host]. It has one compute queue and no copy queue: copies
    of Metal memory are the host's. Its commands are:
    - [exec call prg], which lays out [prg]'s arguments at the next 256-byte
      boundary of the placeholder, the addresses of its buffers on the device
      then its variables, and adds a command running [prg] on them. A launch
      size that reads a variable is written after the arguments, and the host
      program sets it on the command on each run;
    - [wait], which encodes nothing: the device's fence orders its command
      buffers;
    - [signal word v], which makes [v] the value the device's signaler signals
      once the last command buffer completed;
    - [timestamp], which times each command in a command buffer of its own: the
      host program writes that command buffer, retained, over the first of the
      command's two stamps and [0] over the second, once it released the command
      buffer that an earlier run left there unread;
    - [memory_barrier], which encodes nothing;
    - [submit], whose host program encodes the commands into one command buffer,
      or one per command with timestamps. Each command buffer's compute encoder
      waits for the device's fence, declares the device's buffers resident
      unless [residency_set], sets each pipeline once before the [Apple9]
      family, runs its range of the indirect command buffer, updates the fence
      and ends; each is handed to the device's signaler, the last with the value
      and the others with [0], and committed.

    [copy] raises [Invalid_argument]. *)

(** {1:linking What the engine links} *)

val handles : string list
(** [handles] is the names of the words that the ["mtl_sel"] placeholder holds
    first, in order: the device's command queue and its fence ([queue],
    [fence]), the address of the table of its buffers that an encoder declares
    resident without a residency set, and their number ([resources], [count]),
    then its signaler ([signaler]). *)

val selectors : string list
(** [selectors] is the names of the Objective-C selectors whose registered
    values the ["mtl_sel"] placeholder holds after {!handles}, in order. *)

type command = {
  lib : string;  (** The Metal library of the command's program. *)
  name : string;  (** The name of its function in [lib]. *)
  global : int list;  (** Its threadgroups per grid, on three axes. *)
  local : int list;  (** Its threads per threadgroup, on three axes. *)
  offset : int;
      (** The byte offset of its arguments in the placeholder, which is its only
          kernel buffer. *)
}
(** The type for the commands of an indirect command buffer. A launch size that
    reads a variable is [1]: the host program sets it on each run. *)

val icb : Ops.t -> (command list * int) option
(** [icb u] is the commands of the indirect command buffer that the placeholder
    [u] holds, and the byte offset of its header, or [None] if [u] holds none.
*)
