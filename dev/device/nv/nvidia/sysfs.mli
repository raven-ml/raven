(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A machine's NVIDIA GPUs, from its PCI functions' files (private). *)

val gpus : string -> string list
(** [gpus root] is the bus addresses of the NVIDIA GPUs ({!Device_nv.is_gpu})
    among the functions under [root/sys/bus/pci/devices], in bus order: by
    domain, bus, device and function. It is [[]] if the directory does not
    exist; a function whose [vendor] or [class] file cannot be read is none. *)
