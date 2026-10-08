(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A machine's NVIDIA GPUs, from its PCI functions' files. *)

val gpus : string -> string list
(** [gpus root] is the bus addresses of the NVIDIA GPUs ({!Rig_nv.is_gpu})
    among the functions under [root/sys/bus/pci/devices], in bus order: by
    domain, bus, device and function. It is [[]] if the directory does not
    exist; a function without a [vendor] or [class] file is none.

    Raises [Failure] with the system's error if the directory or such a file
    exists and cannot be read, such as when the process has no file left. *)
