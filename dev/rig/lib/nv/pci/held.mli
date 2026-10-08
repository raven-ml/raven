(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The device file NVIDIA's kernel driver serves a GPU through (private).

    NVIDIA's kernel driver gives each GPU a file [/dev/nvidiaN], [N] the GPU's
    minor, which [/proc/driver/nvidia/gpus/<bus>/information] states on its line
    [Device Minor:]. While a process holds that file open, unbinding the driver
    from the GPU waits for the process to close it, so a detach refuses a GPU
    whose file this process holds ({!Rig_pci.Gpus.make}'s [nodes]). *)

val nodes : read:(string -> string option) -> string -> string list
(** [nodes ~read bus] is [["dev/nvidiaN"]], the device file of the GPU at [bus]
    as a path from the machine's root, [N] the minor that
    [read ("proc/driver/nvidia/gpus/" ^ bus ^ "/information")] states. It is
    [[]] if [read] gives [None], as when the kernel driver does not serve the
    GPU, or if the file states no minor. *)
