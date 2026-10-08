(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thread traces: what a GPU's shader engines record while they run waves.

    While tracing, each shader engine writes a stream of packets into a buffer
    of its own: when each wave starts and ends, the instructions it issues and,
    from time to time on GFX11 on, markers of the GPU's clock ({!Pm4.source}).
    Its times count the engine's cycles from the start of the trace.

    A compute queue starts tracing with {!val-start} and stops with {!val-stop};
    the host then reads each engine's buffer up to its {!length} and decodes it
    with {!waves} and {!clock}. The program is the one Mesa writes for a compute
    queue, register by register. Engines are numbered across dies: engine [e] is
    engine [e mod g.shader_engines] of die [e / g.shader_engines], for
    [0 <= e < g.shader_engines * g.xccs].

    The programs take the GPU's work-group processors that run work as
    {!Capability.t}'s [wgps] gives them: [wgps.(e).(a)] for shader array [a] of
    engine [e]. An engine whose arrays are all [0], harvested, is neither traced
    nor awaited, and its end is not stored. *)

(** {1:recording Recording} *)

val start :
  Gpu.t -> wgps:int array array -> size:int -> (int -> 'v) -> 'v Packet.t
(** [start g ~wgps ~size buffer] starts tracing on every shader engine of [g]
    whose arrays run work ([wgps]), engine [e] writing at most [size] bytes into
    its buffer at address [buffer e]. Each
    engine traces the compute waves of one unit of its first shader array: every
    SIMD of its first compute unit on GFX9, the first SIMD of its first
    workgroup processor on GFX11 on. Engines [0] and [1] alone trace the
    instructions the waves issue. It makes the caches coherent before and after.

    [size] and every [buffer e] are multiples of 4096. Raises [Invalid_argument]
    if [size] is not a multiple of 4096 from 4096 to 2{^ 22} - 1 pages of 4096
    bytes, the most an engine's 22-bit size field holds, or if [wgps] does not
    hold [g.shader_engines * g.xccs] engines. *)

val stop : Gpu.t -> wgps:int array array -> (int -> 'v) -> 'v Packet.t
(** [stop g ~wgps ends] stops tracing on every shader engine of [g] whose
    arrays run work ([wgps]), waits until each has written its trace out, then
    stores where engine [e]'s trace ends, a 32-bit word, to address [ends e],
    confirmed before the next packet starts. {!length} reads that word. Raises [Invalid_argument]
    if [wgps] does not hold [g.shader_engines * g.xccs] engines. *)

val length : Gpu.t -> buffer:int -> int -> int
(** [length g ~buffer w] is the bytes a shader engine of [g] wrote into its
    buffer at address [buffer], from the word [w] {!val-stop} stored for it. A
    result outside \[[0];[size]\], [size] as {!val-start} took it, means the
    trace is not whole. *)

(** {1:decoding Decoding} *)

type wave = {
  cu : int;
      (** The compute unit, numbered within its shader engine; on GFX11 on, its
          workgroup processor, with its shader array above it: the array's
          number shifted left by 3 bits on GFX11 and 4 on GFX12. *)
  simd : int;  (** The SIMD of the compute unit. *)
  slot : int;  (** The SIMD's wave slot. *)
  start : int;  (** When the wave started, in shader cycles. *)
  stop : int;  (** When it ended, in shader cycles. *)
}
(** The type for the waves of a trace. *)

val waves : Gpu.t -> string -> wave list
(** [waves g trace] is the waves that start and end in [trace], one shader
    engine's bytes as [g] wrote them, in the order they end. A wave's end is
    paired with the latest start before it of the same compute unit, SIMD and
    slot that no end took; a start or an end without its pair is no wave. A
    trace cut short yields the waves of its whole packets. *)

val clock : Gpu.t -> string -> (int -> int) option
(** [clock g trace] maps a shader time of [trace] to the GPU's clock
    ({!Pm4.source}) through [trace]'s realtime markers: on the line through the
    two markers around it, or through the first two or the last two outside
    them. It is [None] for a trace of fewer than two markers at distinct shader
    times, such as any trace of a GFX9 GPU. *)
