(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs of this machine, through NVIDIA's kernel driver.

    {!open_} opens a GPU that NVIDIA's Linux kernel driver holds as a
    {!Rig_nv.t}, through the driver's resource manager ([/dev/nvidiactl] and the
    GPU's [/dev/nvidiaN]) and its unified memory driver ([/dev/nvidia-uvm]),
    which maps memory into the GPU's address space. The kernel driver owns the
    GPU and shares it with other programs; it alone reports the GPU's faults,
    and lets work run as long as it takes.

    {b Numbering.} GPU [i] is the [i]th of the machine's NVIDIA GPUs
    ({!Rig_nv.is_gpu}) in bus order, whichever kernel driver holds each: a GPU
    another driver holds keeps its index, and does not open here. {!buses}
    lists them.

    {b Releases.} The kernel driver's releases 570, 580, 610 and 615 open; a GPU
    of another release answers [Error] naming them.

    {b The machine's files.} {!buses}, {!count} and {!open_} read this
    machine's files under the directory [root] (defaults to ["/"]): its PCI
    functions under [root/sys/bus/pci] and the kernel driver's files under
    [root/dev], such as a container's view of its host. Every root shows the
    same kernel driver.

    {b The process's client.} The process has one client of the kernel
    driver: three files and its reservation of the GPUs' addresses. The first
    open that makes it, from that open's [root/dev], keeps it for every later
    open, failed or not, whatever its [root]. An open that fails to make it
    keeps nothing.

    {b Memory.} The process's GPU memory lies at addresses below [2{^40}] that
    this library reserves in the process at its first open, from [384 GiB] up.
    Memory the library allocates for the host lies at the same address for the
    host and the GPU; host memory [Rig_nv.map_host] maps lies at its own GPU
    address, as the host's may lie higher. [Mapped] memory ({!Rig_nv.alloc})
    lies in the GPU's BAR1, which the kernel driver sizes, often at 256 MiB.

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

val buses : ?root:string -> unit -> string list
(** [buses ~root ()] is the bus addresses of this machine's NVIDIA GPUs
    ({!Rig_nv.is_gpu}) under [root] (defaults to ["/"]), in bus order: GPU
    [i] is the [i]th, whichever kernel driver holds it. It is [[]] where
    [root/sys/bus/pci] does not exist, such as off Linux. Listing them
    changes nothing on the machine.

    Raises [Failure] with the system's error if the machine's PCI functions
    cannot be read, such as when the process has no file left. *)

val count : ?root:string -> unit -> int
(** [count ~root ()] is [List.length (buses ~root ())], the indices [0] to
    [count ~root () - 1].

    Raises [Failure] as {!buses}. *)

val device_name : int -> string
(** [device_name i] is the name of GPU [i]: ["NV"] for [0], ["NV:i"] otherwise.

    Raises [Invalid_argument] if [i < 0]. *)

val open_ : ?root:string -> int -> (Rig_nv.t, string) result
(** [open_ ~root i] opens GPU [i] under [root] (defaults to ["/"]) through
    NVIDIA's kernel driver. It changes nothing on the machine but the
    kernel driver's state for this process.

    The result is [Error msg] if [i >= count ~root ()], saying how many GPUs
    there are, if the machine's PCI functions cannot be read, saying why,
    if the kernel driver does not hold GPU [i] or cannot be opened, if its
    release is not one this library opens, while a device of GPU [i] is
    open and not stopped, or with the kernel driver's refusal. A GPU has
    one device at a time: the unified memory driver registers a GPU once
    per process. A failed open gives back what it took for GPU [i].

    Raises [Invalid_argument] if [i < 0]. *)
