(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The device file [/dev/nvidiaN] through which NVIDIA's kernel driver
    serves a GPU, which a detach must find closed ({!Rig_pci.Driver.make}'s
    [nodes]). *)

val nodes : root:string -> string -> string list
(** [nodes ~root bus] is [["dev/nvidiaN"]], the device file of the GPU at [bus]
    as a path from the machine's root [root], [N] the minor that
    [proc/driver/nvidia/gpus/BUS/information] under [root] states. It is [[]] if
    that file cannot be read, as when the kernel driver does not serve the GPU,
    or if it states no minor. *)
