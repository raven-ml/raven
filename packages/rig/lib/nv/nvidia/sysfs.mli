(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A machine's NVIDIA GPUs, from its PCI functions' files. *)

val known : string -> string -> (unit, string) result
(** [known root bus] is [Ok ()] iff [bus] is a bus address as Linux spells
    it, ["DDDD:BB:DD.F"] in lowercase hexadecimal with a domain of four
    digits or more, of a function under [root/sys/bus/pci/devices], and
    [Error why], [why] naming [bus], otherwise. *)

val gpus : string -> (string list, string) result
(** [gpus root] is the bus addresses of the NVIDIA GPUs ({!Rig_nv.is_gpu})
    among the functions under [root/sys/bus/pci/devices], in bus order: by
    domain, bus, device and function, each as a number. It is [Ok []] if the
    directory does not exist. A function whose [vendor] or [class] file
    cannot be read, as while the kernel removes it, is left out.

    [Error] with the system's error if the directory exists and cannot be
    read, such as when the process has no file left. *)
