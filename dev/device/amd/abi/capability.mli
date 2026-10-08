(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from an AMD device.

    Code compiled for an AMD device encodes its work with this library's
    encoders for the device's GPU, and gives its kernels scratch memory
    ({!Scratch}). The device's driver gives what this needs as a {!t} when it
    opens the device, under {!key}.

    {b Times.} The GPU writes times ({!Pm4.copy_data} of {!Pm4.Clock},
    {!Sdma.timestamp}, the markers of a {!Thread_trace}) as counts of its clock,
    which runs at {!field-clock_hz}. A reader converts them to the host clock
    ([CLOCK_MONOTONIC]) by that frequency and an offset, measured with a time
    the GPU writes between two readings of the host clock.

    {b Fills.} Work for one of the device's queues can be a C function, a
    {e fill}: [int fill(void *queue, void *arg, uint64_t v)]. [queue] is the
    queue's writer, valid only during the call, and [v] the value the work
    completes on the device's timeline. The fill places its packets with
    {!field-place} and takes memory for its kernels' arguments with
    {!field-segment}, and nothing else: waiting for earlier work and signalling
    [v] are the driver's. It stops at the first call that fails and returns the
    failure that call returned, or returns [0] once every call succeeded.

    A fill whose work declares [r] ring units places at most [r] words
    ({!Packet.size}), and one whose work declares [s] segment bytes takes at
    most [s] bytes: a call beyond them returns a failure. *)

(** The type for the packets a device's compute queue reads. *)
type compute =
  | Pm4
      (** PM4 packets ({!Pm4}), which one die runs. Each dispatch names its
          scratch buffer ({!Pm4.dispatch}). *)
  | Aql of { scratch : int -> (unit, string) result }
      (** AQL packets ({!Aql}), which every die runs. The queue hands every
          kernel it dispatches one scratch buffer, the driver's. [scratch n]
          makes it serve kernels of up to [n] bytes per lane ({!Scratch.size}):
          it does nothing if the buffer already does, else replaces it with a
          larger one, kept until the work placed before the call completes. The
          result is [Error msg] if the device cannot allocate it. Any domain may
          call it. *)

type trace = {
  buffers : int;  (** The GPU address of the trace windows. *)
  buffers_host : nativeint;  (** Their host address. *)
  window : int;  (** The bytes of a run's trace of one shader engine. *)
  slots : int;  (** The runs the buffers hold. *)
  engines : int;  (** The shader engines, numbered across dies. *)
  ends : int;  (** The GPU address of the end words. *)
  ends_host : nativeint;  (** Their host address. *)
}
(** The type for the device's thread trace buffers: one set per GPU, which every
    program that traces writes ({!Thread_trace}). Each of [slots] runs traces
    each shader engine into a window of its own: shader engine [e] of the run in
    slot [s] writes the [window] bytes at
    [buffers + ((e * slots) + s) * window], and its end at the 32-bit word at
    [ends + 4 * ((s * engines) + e)] ({!Thread_trace.stop}). Two runs whose
    traces have not been read take two slots: the programs that trace share the
    slots, and the reader reads a slot's trace before a later run takes it. *)

type t = {
  gpu : Gpu.t;  (** The device's GPU, as the encoders take it. *)
  clock_hz : int;  (** The frequency of the GPU's clock, in hertz. *)
  compute : compute;  (** What its compute queue reads. *)
  place : nativeint;
      (** [place] is the address of
          [int place(void *queue, const uint32_t *words, size_t n)], which a
          fill calls to place the [n] words at [words] on its queue: whole
          packets of the queue's kind. The writer applies the ring's rules: a
          PM4 ring wraps, an SDMA or AQL [words] goes whole before the ring's
          end, and an AQL packet's first word is stored last ({!Aql}). [n] is a
          multiple of 16 on an AQL queue.

          It returns [0], or a failure that the fill returns as its own: a
          failure if the words would pass the ring units its work declared. Only
          a fill, during its call, may call it. *)
  segment : nativeint;
      (** [segment] is the address of
          [int segment(void *queue, size_t n, void **host, uint64_t *address)],
          which a fill calls to take [n] bytes of the device's memory for its
          kernels' arguments, such as a dispatch packet a kernel reads
          ({!Aql.dispatch}). It stores at [host] where the host writes them and
          at [address] where the GPU reads them, a multiple of 64. They stay
          valid until [v] is reached. The driver makes what the host wrote there
          visible to the GPU before it hands the work over.

          It returns [0], or a failure that the fill returns as its own: a
          failure if the bytes would pass the segment bytes its work declared.
          Only a fill, during its call, may call it. *)
  wgps : int array array;
      (** The work-group processors that run work: [wgps.(e).(a)] has bit [w]
          set iff processor [w] of shader array [a] of shader engine [e] does,
          engines numbered across dies. A counter's value for a processor that
          runs no work stays [0] ({!Counter}). *)
  trace : unit -> (trace, string) result;
      (** [trace ()] is the device's trace buffers, made at the first call and
          the same at every later one. The first call also holds the GPU's
          clocks and shader engines steady for the rest of the process, as
          tracing needs. The result is [Error msg] if the device has not the
          memory, or cannot hold its GPU steady, such as when another process
          holds it so. Any domain may call it. *)
}
(** The type for what compiled code needs from an AMD device. *)

val key : t Type.Id.t
(** [key] is the key an AMD device's {!t} is found under. *)
