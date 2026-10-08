(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The files of a machine the amdgpu path reads, under its root directory ("/"
   for this machine): its PCI functions, the kernel driver's topology of its
   GPUs, and the versions of their blocks. Pure reads: any domain may call
   them. *)

(* A GPU as the kernel driver's topology describes it. *)
type node = {
  index : int; (* its node in the topology *)
  gpu_id : int; (* the kernel driver's identifier of the GPU *)
  render : int; (* the minor of its render node, /dev/dri/renderD<render> *)
  gpu : Device_amd_abi.Gpu.t;
  lds : int; (* a workgroup's local data share, in bytes *)
  mec : int; (* the version of its compute queues' firmware *)
  budget : int; (* its memory, in bytes *)
  waves_per_cu : int; (* the most waves a compute unit runs at once *)
  arrays : int; (* the shader arrays of a shader engine *)
  cwsr : int; (* a die's context save area, in bytes *)
  ctl_stack : int; (* the part of it that holds the control stack *)
}

(* [gpus root] is the bus addresses of the AMD GPUs among the PCI functions
   under [root], in bus order. *)
val gpus : string -> string list

(* [node root bus] is the GPU at bus address [bus], or why the kernel driver
   describes none there. *)
val node : string -> string -> (node, string) result

(* [linked root n n'] is [true] iff the topology lists a link from node [n] to
   node [n']: over XGMI, or over PCIe through a large BAR. *)
val linked : string -> int -> int -> bool
