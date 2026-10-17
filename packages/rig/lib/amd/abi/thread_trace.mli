(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thread traces, which a GPU's shader engines record while they run
    waves.

    While tracing, each shader engine writes a stream of packets into a
    buffer of its own: when each wave starts and ends, the instructions it
    issues and, from GFX11 on, markers of the GPU's clock ({!Pm4.source})
    from time to time. Its times count the engine's cycles from the start of
    the trace.

    A compute queue starts tracing with the PM4 packets of {!val-start} and
    stops with those of {!val-stop}, which a submission places on the queue
    ([Rig.Submission.words]). The host then reads each engine's buffer up
    to its {!length} and decodes it with {!waves} and {!clock}. Engines are
    numbered across dies: engine [e] is engine [e mod g.shader_engines] of
    die [e / g.shader_engines], for [0 <= e < g.shader_engines * g.xccs].
    Addresses are as the device's work addresses memory
    ([Rig.Submission.address]).

    The programs take the GPU's work-group processors that run work as
    {!Capability.t}'s [wgps] gives them: [wgps.(e).(a)] for shader array [a]
    of engine [e]. An engine whose arrays are all [0] is harvested: it is
    neither traced nor awaited, and its end is not stored. *)

(** {1:recording Recording} *)

val start :
  Gpu.t -> wgps:int array array -> size:int -> address:int -> string
(** [start g ~wgps ~size ~address] is the packets, as little-endian 32-bit
    words, that start tracing on every shader engine of [g] whose arrays run
    work ([wgps]), engine [e] writing at most [size] bytes into its buffer
    at [address + (e * size)]. They make the caches coherent before and
    after.
    {ul
    {- Each engine traces the compute waves of one unit of its first shader
       array: every SIMD of its first compute unit on GFX9, and the first
       SIMD of its first work-group processor from GFX11 on.}
    {- Engines [0] and [1] alone trace the instructions the waves issue.}}

    Raises [Invalid_argument] if [size] is not a multiple of 4096 from 4096
    to [2{^22} - 1] pages of 4096 bytes, the most an engine's 22-bit size
    field holds, if [address] is not a non-negative multiple of 4096, if
    [wgps] does not hold [g.shader_engines * g.xccs] engines, or if [g]'s
    GC lacks a register the program writes. *)

val stop : Gpu.t -> wgps:int array array -> address:int -> string
(** [stop g ~wgps ~address] is the packets, as little-endian 32-bit words,
    that stop tracing on every shader engine of [g] whose arrays run work
    ([wgps]) and wait until each has written its trace out. They then store
    where engine [e]'s trace ends, a 32-bit word, at [address + (4 * e)],
    confirmed before the next packet starts. {!length} reads that word.

    Raises [Invalid_argument] if [address] is not a non-negative multiple
    of 4, if [wgps] does not hold [g.shader_engines * g.xccs] engines, or if
    [g]'s GC lacks a register the program reads or writes. *)

val length : Gpu.t -> buffer:int -> int -> int
(** [length g ~buffer w] is the bytes a shader engine of [g] wrote into its
    buffer at address [buffer], from the word [w] {!val-stop} stored for
    it. A result outside \[[0];[size]\], [size] as {!val-start} took it,
    means the trace is not whole. *)

(** {1:decoding Decoding} *)

type wave = {
  cu : int;
      (** The compute unit, numbered within its shader engine. From GFX11
          on, it is the work-group processor, with its shader array above
          it: the array's number shifted left by 3 bits on GFX11 and 4 on
          GFX12. *)
  simd : int;  (** The SIMD of the compute unit. *)
  slot : int;  (** The SIMD's wave slot. *)
  start : int;  (** When the wave started, in shader cycles. *)
  stop : int;  (** When it ended, in shader cycles. *)
}
(** The type for the waves of a trace. *)

val waves : Gpu.t -> string -> wave list
(** [waves g trace] is the waves that start and end in [trace], one shader
    engine's bytes as [g] wrote them, in the order they end. A wave's end is
    paired with the latest start before it of the same compute unit, SIMD
    and slot, unless an end of that slot lies between them.
    {ul
    {- A slot runs one wave at a time, so a second start before an end
       means the trace lost the first wave's end, and that first start is
       dropped.}
    {- An end without a start is dropped.}
    {- A trace cut short yields the waves of its whole packets.}} *)

val clock : Gpu.t -> string -> (int -> int) option
(** [clock g trace] maps a shader time of [trace] to the GPU's clock
    ({!Pm4.source}) through [trace]'s realtime markers. A time between two
    markers maps on the line through them, and a time outside them on the
    line through the first two or the last two. It is [None] for a trace of
    fewer than two markers at distinct shader times, such as any trace of a
    GFX9 GPU. *)
