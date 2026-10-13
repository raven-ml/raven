(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory devices: the driver that {!Rig.memory_device} opens.

    A memory device's memory is host memory from the C library's aligned
    allocator, and its work runs in the thread that submits it. Its C
    [submit] runs a submission's fills and copies in order, then stores the
    value in the device's word, so every value is reached when {!Rig.submit}
    returns. Its C [room] refuses any other part for good ([RIG_NEVER]), so
    a submission with a launch or a words part never runs on it.

    Its two queues, [COMPUTE:0] and [COPY:0], each run fills and copies, and
    [COPY:0] is its copy engine. Its budget is [max_int]. It maps every host
    address, and the host and every driver's device on this machine reach
    its memory ({!Dev.reaches}). Its device record is marked as a memory device
    ({!Def.device}), so {!Rig.runs_on_host} is [true] of it.

    Its state is its driver's table and its word, on a line of its own in
    pages that hold such states alone, which other devices may map. It lives
    for the process. *)

val open_ : string -> (Def.device, string) result
(** [open_ name] is {!Rig.memory_device}[ name]. *)
