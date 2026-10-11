(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The files of a machine that the amdgpu path reads under its root
    directory (["/"] for this machine): its PCI functions, the kernel
    driver's topology of its GPUs, and the versions of their blocks. Each
    value only reads files, so any domain may call it. A file that is
    missing or unreadable reads as absent, but for {!gpus}. *)

type node = {
  index : int;  (** Its node in the topology. *)
  gpu_id : int;  (** The kernel driver's identifier of the GPU. *)
  render : int;
      (** The minor of its render node, [/dev/dri/renderD<render>]. *)
  gpu : Rig_amd_abi.Gpu.t;
  lds : int;  (** A workgroup's local data share, in bytes. *)
  mec : int;  (** The version of its compute queues' firmware. *)
  budget : int;  (** Its memory, in bytes. *)
  visible : int;  (** Of it, the bytes the host reaches through the BAR. *)
  waves_per_cu : int;  (** The most waves a compute unit runs concurrently. *)
  arrays : int;  (** The shader arrays of a shader engine. *)
  cu_per_array : int;
      (** The compute units a shader array has, active or not. *)
  cwsr : int;  (** A die's context save area, in bytes. *)
  ctl_stack : int;  (** The part of it that holds the control stack. *)
}
(** The type for a GPU as the kernel driver's topology describes it. *)

val gpus : string -> (string list, string) result
(** [gpus root] is the bus addresses of the AMD GPUs among the PCI functions
    under [root], in bus order: by domain, bus, device and function, each as
    a number. It is [Ok []] where [root/sys/bus/pci/devices] does not exist.
    A function without a [vendor] or [class] file is not a GPU.

    [Error] with the system's error if the directory or such a file exists
    and cannot be read, such as when the process has no file left. *)

val node : string -> string -> (node, string) result
(** [node root bus] is the GPU at bus address [bus], or why the kernel driver
    describes none there. Of a GPU split into partitions, it is the first
    partition's node. *)

val linked : string -> int -> int -> bool
(** [linked root n n'] is [true] iff the topology lists a link from node
    [n] to node [n']: an I/O link over XGMI, or a peer-to-peer link over
    PCIe through a large BAR. *)
