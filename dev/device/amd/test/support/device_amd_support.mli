(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suites and bench share. *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no AMD GPU. The lock
    is [flock] on [/tmp/raven-device-gpu.lock], the file every suite that acts
    on a GPU of the machine locks; its holder writes its executable and process
    id into it. A suite calls [hold_gpu] before [Windtrap.run], so that the wait
    counts against no test's timeout; {!gpu} calls it again.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

val gpu : unit -> Device_amd.t
(** [gpu ()] is AMD GPU [0], opened through the amdgpu path, after stopping the
    device an earlier {!gpu} opened if no {!stop} stopped it, as a failed test
    leaves it, while the process holds the machine's GPU lock ({!hold_gpu}). It
    skips the test if the machine has no AMD GPU. *)

val stop : Device_amd.t -> unit
(** [stop g] is [Device_amd.stop g]. Tests stop the devices {!gpu} opened
    through it. *)

val with_gpu : (Device_amd.t -> 'a) -> 'a
(** [with_gpu f] is [f (gpu ())], the device stopped after. *)

val wait : Device_amd.t -> int -> unit
(** [wait g v] returns once [g]'s timeline word reaches [v], sleeping on the
    device between reads. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of CPU time, and
    asserts under [w] that each read is [x]. *)

val read : int -> int -> string
(** [read a n] is the [n] bytes of host memory at [a]. *)

val write : int -> string -> unit
(** [write a s] writes [s] to host memory at [a]. *)

val pages : int -> int
(** [pages n] is the host address of [n] new zeroed bytes on pages of their
    own. *)

val free_pages : int -> int -> unit
(** [free_pages a n] gives back the [n] bytes at [a] that {!pages} gave. *)

val fill :
  ?code:int ->
  Device_amd.capability ->
  int array ->
  bytes:int ->
  nativeint * nativeint
(** [fill ~code c ws ~bytes] is a fill, as a C function and its argument, that
    places the words [ws] with [c]'s [place], then takes [bytes] bytes of the
    argument segment with [c]'s [segment] if [bytes > 0], and returns the first
    failure of these, else [code] (defaults to [0]). The argument lives as long
    as the process. *)

val room :
  ?words:int ->
  ?fill:bool ->
  ?copy:int ->
  ?after:int ->
  Device_amd.t ->
  queue:int ->
  [ `Fits | `Later | `Never ]
(** [room ~words ~fill ~copy ~after g ~queue] is what [device_amd_room]
    answers for one part on the queue of index [queue]: [words] words (at most
    [64]; defaults to none), a fill if [fill] (defaults to [false]), a copy of
    [copy] bytes if positive (defaults to [0]), after part [after] if it is not
    negative (defaults to [-1]). *)
