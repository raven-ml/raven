(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Software devices: the driver that {!Rig.software} opens.

    A software device's memory is host memory from the C library's aligned
    allocator, and its work runs in the thread that submits it. Its image is
    host code that {!Host_code} links, and its arch is the host's. Its C
    [submit] runs a submission's fills, copies and launches in order, a
    launch as the host runs one, times them on the host's clock, then stores
    the value in the device's word, so every value is reached when
    {!Rig.submit} returns. Its C [room] refuses a words part, and a launch
    the host refuses, for good ([RIG_NEVER]).

    Its two queues are [COMPUTE:0], which runs launches, fills and copies,
    and [COPY:0], its copy engine, which runs fills and copies. Its budget
    is [max_int]. It maps every host address, and the host and every
    driver's device on this machine that maps host memory reach its memory
    ({!Dev.reaches}). Its device record is marked as a software device
    ({!Def.device}), so {!Rig.runs_on_host} is [true] of it.

    Its state is its driver's table and the address of its word, which lies
    where rig gave it to the open. The open allocates the state and the stop
    frees it, after which rig calls none of the driver's functions. *)

val open_ : string -> (Def.device, string) result
(** [open_ name] is {!Rig.software}[ name]. *)
