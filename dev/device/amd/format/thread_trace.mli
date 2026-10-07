(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Thread traces: what a GPU's shader engines record while they run waves.

    While tracing, each shader engine writes a stream of packets into a buffer
    of its own: when each wave starts and ends, the instructions it issues and,
    from time to time on GFX11 on, markers of the GPU's realtime clock. Its
    times count the engine's cycles from the start of the trace.

    A compute queue starts tracing with {!start} and stops with {!stop}; the
    host then reads each engine's buffer up to its {!length} and decodes it with
    {!waves} and {!clock}. Engines are numbered across dies: engine [e] is
    engine [e mod g.shader_engines] of die [e / g.shader_engines], for
    [0 <= e < g.shader_engines * g.xccs]. *)

(** {1:recording Recording} *)

val start : Gpu.t -> size:int -> (int -> 'v) -> 'v Packet.t
(** [start g ~size buffer] starts tracing on every shader engine of [g], engine
    [e] writing at most [size] bytes into its buffer at address [buffer e]. It
    traces compute waves on the first SIMD of each engine's first workgroup
    processor, and the instructions they issue on engines [0] and [1] only. It
    makes the caches coherent before and after.

    [size] and every [buffer e] are multiples of 4096. Raises [Invalid_argument]
    if [size] is not a positive multiple of 4096. *)

val stop : Gpu.t -> (int -> 'v) -> 'v Packet.t
(** [stop g ends] stops tracing on every shader engine of [g], waits until each
    has written its trace out, then stores where engine [e]'s trace ends, a
    32-bit word, to address [ends e], confirmed before the next packet starts.
    {!length} reads that word. *)

val length : Gpu.t -> buffer:int -> int -> int
(** [length g ~buffer w] is the bytes a shader engine of [g] wrote into its
    buffer at address [buffer], from the word [w] {!stop} stored for it. A
    result outside \[[0];[size]\] means the trace is not whole. *)

(** {1:decoding Decoding} *)

type wave = {
  cu : int;
      (** The compute unit, numbered within its shader engine; on GFX11 on, its
          workgroup processor and shader array. *)
  simd : int;  (** The SIMD of the compute unit. *)
  slot : int;  (** The SIMD's wave slot. *)
  start : int;  (** When the wave started, in shader cycles. *)
  stop : int;  (** When it ended, in shader cycles. *)
}
(** The type for the waves of a trace. *)

val waves : Gpu.t -> string -> wave list
(** [waves g trace] is the waves that start and end in [trace], one shader
    engine's bytes as [g] wrote them, in the order they end. A wave's start is
    paired with the next end of the same compute unit, SIMD and slot. A trace
    cut short yields the waves of its whole packets. *)

val clock : Gpu.t -> string -> (int -> int) option
(** [clock g trace] maps a shader time of [trace] to the GPU's 100 MHz realtime
    clock through [trace]'s realtime markers: on the line through the two
    markers around it, or through the first two or the last two outside them. It
    is [None] for a trace of fewer than two markers at distinct shader times,
    such as any trace of a GFX9 GPU. *)
