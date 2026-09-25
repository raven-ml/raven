(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Queue execution profiles. Set [PROFILE=1] before compiling and running a
    workload, then collect its events with {!Device.profile}. *)

type event = {
  device : string;
  queue : string;
  name : string;
  start_us : float;
  duration_us : float;
}
(** A completed operation in microseconds. [start_us] uses the calibrated
    host wall clock; [duration_us] uses the original device-clock difference. *)

val calibrate : (unit -> unit -> float) -> float
(** [calibrate sample] estimates the offset from device microseconds to host
    wall-clock microseconds. [sample ()] enqueues a fresh timestamp and returns
    a function that waits for it and reads device microseconds. Host readings
    bracket that wait, excluding command preparation. The median of five
    samples reduces scheduling outliers. The offset is an
    estimate; it does not establish execution dependencies.
    Raises [Invalid_argument] for a non-finite sample. *)

val output : out_channel -> event list -> unit
(** [output channel events] writes Chrome trace-event JSON to [channel],
    without closing or flushing it. Each device occupies a separate process
    lane, with separate threads for its queues. One shared origin is subtracted
    from every event, preserving calibrated cross-device offsets. Names are
    JSON-escaped.

    Raises [Invalid_argument] if a timestamp or duration is not finite,
    if either is negative, or if names are not valid UTF-8. *)
