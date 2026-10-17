(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What compiled code needs from an AMD device.

    Code compiled for an AMD device encodes its work with this library's
    encoders for the device's GPU, and gives its kernels scratch memory
    ({!Scratch}). The device's driver declares these as a {!t} when it opens
    the device ({!declare}), and compiled code finds them under {!key}.

    {b Times.} The GPU writes times ({!Pm4.copy_data} of {!Pm4.Clock},
    {!Sdma.timestamp}, the markers of a {!Thread_trace}) as counts of its
    clock, which runs at {!field-clock_hz}. A reader converts them to the
    host clock ([CLOCK_MONOTONIC]) by that frequency and an offset, measured
    with a time the GPU writes between two readings of the host clock. *)

(** The type for the packets a device's compute queue reads. *)
type compute =
  | Pm4
      (** PM4 packets ({!module-Pm4}), which one die runs. Each dispatch
          names its scratch buffer ({!Pm4.dispatch}). *)
  | Aql of { scratch : int -> (unit, string) result }
      (** AQL packets ({!module-Aql}), which every die runs. The queue hands
          every kernel it dispatches one scratch buffer, the driver's.
          [scratch n] makes it serve kernels of up to [n] bytes per lane
          ({!Scratch.size}). It does nothing if the buffer already does, and
          otherwise replaces it with a larger one, keeping the old one until
          the work placed before the call completes. Any domain may call
          it. The result is [Error msg] if the device cannot allocate the
          buffer. Raises [Invalid_argument] as {!Scratch.descriptor} or
          {!Scratch.tmpring} does for it, and, having changed nothing, from
          a callback inside a lock ([Rig]'s domains section). *)

type trace = {
  buffers : int;  (** The GPU address of the trace windows. *)
  buffers_host : int;  (** Their host address. *)
  window : int;  (** The bytes of a run's trace of one shader engine. *)
  slots : int;  (** The runs the buffers hold. *)
  engines : int;  (** The shader engines, numbered across dies. *)
  ends : int;  (** The GPU address of the end words. *)
  ends_host : int;  (** Their host address. *)
}
(** The type for the device's thread trace buffers, one set per GPU, which
    every program that traces writes ({!Thread_trace}). The buffers hold
    [slots] runs, and each run traces each shader engine into a window of
    its own. Shader engine [e] of the run in slot [s] writes the [window]
    bytes at [buffers + ((e * slots) + s) * window], and its end at the
    32-bit word at [ends + 4 * ((s * engines) + e)]
    ({!Thread_trace.val-stop}). The programs that trace share the slots: a
    run whose trace has not been read holds its slot, and the reader must
    read a slot's trace before a later run takes it. *)

type t = {
  gpu : Gpu.t;  (** The device's GPU, as the encoders take it. *)
  clock_hz : int;  (** The frequency of the GPU's clock, in hertz. *)
  compute : compute;  (** What its compute queue reads. *)
  wgps : int array array;
      (** The work-group processors that run work: [wgps.(e).(a)] has bit
          [w] set iff processor [w] of shader array [a] of shader engine [e]
          does, engines numbered across dies. On a die whose processors the
          driver cannot read, every processor of its arrays is set. A
          counter's value for a processor that runs no work stays [0]
          ({!Counter}). *)
  trace : unit -> (trace, string) result;
      (** [trace ()] is the device's trace buffers, made at the first call
          and the same at every later one. The first call also holds the
          GPU's clocks and shader engines steady for the rest of the process,
          as tracing needs. Any domain may call it. The result is
          [Error msg] if the device lacks the memory, cannot hold its GPU
          steady, as when another process holds the GPU so, or is
          stopped. Raises [Invalid_argument], having changed nothing, from a
          callback inside a lock ([Rig]'s domains section). *)
}
(** The type for what compiled code needs from an AMD device. *)

val key : t Type.Id.t
(** [key] is the key of an AMD device's record. Compiled code finds the record
    of a device [d] with [Rig.capability d key], which is [None] if [d] is
    not an AMD device. *)

val declare : t -> Rig_edge.capability
(** [declare c] is [c] under {!key}, as the facts of an AMD device carry it
    ({!Rig_edge.type-facts}). *)
