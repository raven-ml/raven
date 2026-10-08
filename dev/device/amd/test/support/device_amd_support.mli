(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suites and bench share. *)

val gpu_lock : string
(** [gpu_lock] is the environment variable that names the machine's GPU lock
    file: ["DEVICE_AMD_TEST_GPU_LOCK"]. *)

val gpu : unit -> Device_amd.t
(** [gpu ()] is AMD GPU [0], opened through the amdgpu path, after stopping the
    device an earlier {!gpu} opened if no {!stop} stopped it, as a failed test
    leaves it. It skips the test if the machine has no AMD GPU, if {!gpu_lock}
    names no file, or if another process holds the lock, which this process
    keeps until it exits once it took it. *)

val stop : Device_amd.t -> [ `Stopped | `Unknown ]
(** [stop g] is [Device_amd.stop g]. Tests stop the devices {!gpu} opened
    through it. *)

val with_gpu : (Device_amd.t -> 'a) -> 'a
(** [with_gpu f] is [f (gpu ())], the device stopped after. *)

val wait : Device_amd.t -> int -> unit
(** [wait g v] returns once [g]'s timeline word reaches [v], sleeping on the
    device between reads. *)

val read : nativeint -> int -> string
(** [read a n] is the [n] bytes of host memory at [a]. *)

val write : nativeint -> string -> unit
(** [write a s] writes [s] to host memory at [a]. *)
