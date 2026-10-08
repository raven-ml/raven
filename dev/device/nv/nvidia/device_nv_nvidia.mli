(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs of this machine, through NVIDIA's kernel driver.

    Opens a GPU that NVIDIA's Linux kernel driver holds as a {!Device_nv.t}:
    through the driver's resource manager ([/dev/nvidiactl] and the GPU's
    [/dev/nvidiaN]) and its unified memory driver ([/dev/nvidia-uvm]), which
    maps memory into the GPU's address space. The kernel driver owns the GPU and
    shares it with other programs; it alone reports the GPU's faults, and lets
    work run as long as it takes.

    {b Numbering.} GPU [i] is the [i]th of the machine's NVIDIA GPUs
    ({!Device_nv.is_gpu}) in bus order, whichever kernel driver holds each: a
    GPU another driver holds keeps its index, and does not open here.

    {b Releases.} The kernel driver's releases 570, 580, 610 and 615 open; a GPU
    of another release answers [Error] naming them.

    {b Memory.} The process's GPU memory lies at addresses below [2{^40}] that
    this library reserves in the process at its first open, from [64 GiB] up:
    memory the host addresses lies at the same address for the host and the GPU.
    The memory [`Mapped] gives the host through the GPU's BAR1, which the kernel
    driver sizes, often at 256 MiB.

    {b Domains.} Any domain may call any function, at the same time as others.

    {b References.}
    - NVIDIA's
      {{:https://github.com/NVIDIA/open-gpu-kernel-modules}
       open-gpu-kernel-modules}, releases 570.144, 580, 610 and 615:
      [kernel-open/common/inc/nv-ioctl-numbers.h] and [nv-ioctl.h] (the escape
      ioctls), [src/nvidia/arch/nvalloc/unix/include/nv_escape.h],
      [src/common/sdk/nvidia/inc/nvos.h] ([NVOS00], [NVOS02], [NVOS21],
      [NVOS32], [NVOS33], [NVOS46], [NVOS54]), [ctrl0000gpu.h], [ctrl0080gpu.h],
      [ctrl2080fb.h], [ctrl2080gpu.h], [ctrl2080gr.h], and
      [kernel-open/nvidia-uvm/uvm_ioctl.h] and [uvm_linux_ioctl.h].
    - Linux's [Documentation/ABI/testing/sysfs-bus-pci]: a function's [vendor]
      and [class] files. *)

val count : unit -> int
(** [count ()] is the number of NVIDIA GPUs of this machine: [0] where the
    machine has no [/sys/bus/pci], such as off Linux. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["NV"] for [0], ["NV:i"] otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ : int -> (Device_nv.t, string) result
(** [open_ i] opens GPU [i] through NVIDIA's kernel driver. It changes nothing
    on the machine but the kernel driver's state for this process.

    The result is [Error msg] if [i >= count ()], saying how many GPUs there
    are, if the kernel driver does not hold GPU [i] or cannot be opened, if its
    release is not one this library opens, while a device of GPU [i] is open and
    not stopped, or with the kernel driver's refusal. A GPU has one device at a
    time: the unified memory driver registers a GPU once per process. A failed
    open gives back what it took.

    Raises [Invalid_argument] if [i < 0]. *)

(**/**)

(* [gpus_at root] is the bus addresses of the NVIDIA GPUs of the machine whose
   files are under the directory [root], in bus order: [/] for this machine, a
   fixture's directory in tests. *)
val gpus_at : string -> string list
