(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The device file NVIDIA's kernel driver serves a GPU through.

    NVIDIA's kernel driver gives each GPU a file [/dev/nvidiaN], [N] the GPU's
    minor, which [/proc/driver/nvidia/gpus/<bus>/information] states on its line
    [Device Minor:]. While a process holds that file open, unbinding the driver
    from the GPU waits for the process to close it, so a detach refuses a GPU
    whose file this process holds ({!Rig_pci.Gpus.make}'s [nodes]). *)

val nodes : root:string -> string -> string list
(** [nodes ~root bus] is [["dev/nvidiaN"]], the device file of the GPU at [bus]
    as a path from the machine's root [root], [N] the minor that
    [proc/driver/nvidia/gpus/BUS/information] under [root] states. It is [[]] if
    that file cannot be read, as when the kernel driver does not serve the GPU,
    or if it states no minor. *)
