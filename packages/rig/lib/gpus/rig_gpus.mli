(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPUs of this machine.

    A {e path} is one way rig reaches a vendor's GPUs, a library of its own:
    through the vendor's runtime, through its kernel driver, or through no
    driver. One card is a GPU of each path that sees it, and its PCI bus
    address says which GPUs of the paths are one card. A path lists its GPUs
    only when asked, since listing loads the vendor's library or reads the
    system's devices.

    A program that takes whichever GPU is here opens the first that opens
    ({!first}):
    {[
    let gpu = Rig_gpus.first ()
    ]} *)

type gpu = private {
  name : string;  (** Its device's name once open, such as ["CUDA:1"]. *)
  bus : string option;
      (** Its PCI bus address, such as ["0000:01:00.0"], the card's identity
          across paths, where its path names one. Metal's GPUs have none. *)
  open_ : unit -> (Rig.t, string) result;
      (** Opens the GPU with {!Rig.open_}. *)
}
(** The type for a GPU a path sees. *)

type path = private {
  kind : string;
      (** Its name, which is its GPU [0]'s device's, such as ["CUDA"]. *)
  gpus : unit -> (gpu list, string) result;
      (** Lists its GPUs on this machine in its own order, or [Error why] if
          it cannot. Each call lists them again and opens no GPU. *)
}
(** The type for a path of rig. *)

val paths : ?firmware:string list -> unit -> path list
(** [paths ~firmware ()] is each path of rig. It lists no GPU. [firmware]
    (defaults to [[]]) is where the driver-less paths look for firmware
    first ({!Rig_nv_pci.gpu}, {!Rig_amd_pci.gpu}).

    Paths come in this order:
    + ["METAL"] ({!Rig_metal}) and ["CUDA"] ({!Rig_cuda}), through the vendor's
      runtime;
    + ["NV"] ({!Rig_nv_nvidia}) and ["AMD"] ({!Rig_amd_amdgpu}), through the
      vendor's kernel driver;
    + ["NV-PCI"] ({!Rig_nv_pci}) and ["AMD-PCI"] ({!Rig_amd_pci}), through no
      driver: the GPUs no kernel driver holds.

    The vendor's runtime comes first, then its kernel driver, then no driver:
    each step down, rig does more of the vendor's work itself. One card
    appears under each path that sees it, with one [bus]. A GPU that a kernel
    driver holds appears under its runtime path and its kernel path, and
    under its driver-less path, whose open then fails. A program that uses
    every card opens one GPU per [bus]. Opening has a cost and takes the
    GPU, so a program opens only the GPUs it uses.

    A path's [gpus] raises [Invalid_argument] as {!Rig_cuda.buses} does, from
    a callback inside a lock ([Rig]'s domains section). *)

val first : ?firmware:string list -> unit -> (Rig.t, string) result
(** [first ~firmware ()] is the first GPU of {!paths} that opens, or
    [Error why], where [why] names each path's refusal and each GPU's. It
    lists a path only once every path before it has opened no GPU.

    Raises [Invalid_argument] as a path's [gpus] does. *)
