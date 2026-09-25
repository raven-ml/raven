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
(** A completed operation measured in microseconds on its device's clock.
    Clocks on different devices are not calibrated against each other. *)

val output : out_channel -> event list -> unit
(** [output channel events] writes Chrome trace-event JSON to [channel],
    without closing or flushing it. Each device occupies a separate process
    lane, with separate threads for its queues and its earliest event at zero;
    cross-device offsets are not
    execution dependencies. Names are JSON-escaped.

    Raises [Invalid_argument] if a timestamp or duration is not finite,
    if either is negative, or if names are not valid UTF-8. *)
