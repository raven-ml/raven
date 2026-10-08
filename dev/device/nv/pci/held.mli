(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Whether this process holds a GPU's kernel-driver file open (private).

    NVIDIA's kernel driver gives each GPU a file [/dev/nvidiaN], [N] the GPU's
    minor, which [/proc/driver/nvidia/gpus/<bus>/information] states on its line
    [Device Minor:]. While a process holds that file open, unbinding the driver
    from the GPU waits for the process to close it. The files are read under a
    root directory: [/] for this machine, a fixture's directory in tests. *)

val nvidia : root:string -> string -> (bool, string) result
(** [nvidia ~root bus] is [true] iff a link under [root/proc/self/fd] names
    [/dev/nvidiaN], [N] the minor of the GPU at [bus]. It is [false] if the
    driver states no minor for [bus], as when it does not hold the GPU, and
    [Error] naming the file if a file the driver states cannot be read. *)
