(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The GPUs of this machine.

    A program that takes whichever GPU is here opens the first of {!openers}
    that opens:
    {[
    let gpu =
      List.find_map
        (fun (_, o) -> Result.to_option (o ()))
        (Rig_gpus.openers ())
    ]} *)

val openers : unit -> (string * (unit -> (Rig.t, string) result)) list
(** [openers ()] is, for each GPU a path of rig sees on this machine,
    the path's name and a function that opens the GPU with {!Rig.open_}. A
    path's name is the name of its GPU [0]'s device, such as ["CUDA"].

    Paths come in this order, each with its GPUs in its own order:
    + ["METAL"] ({!Rig_metal}) and ["CUDA"] ({!Rig_cuda}), through the vendor's
      runtime;
    + ["NV"] ({!Rig_nv_nvidia}) and ["AMD"] ({!Rig_amd_amdgpu}), through the
      vendor's kernel driver;
    + ["NV-PCI"] ({!Rig_nv_pci}) and ["AMD-PCI"] ({!Rig_amd_pci}), through no
      driver: the GPUs no kernel driver holds, booted with firmware from
      rig's firmware directory or [/lib/firmware] ({!Rig_nv_pci.gpu},
      {!Rig_amd_pci.gpu}).

    The vendor's runtime comes first, then its kernel driver, then no driver:
    each step down, rig does more of the vendor's work itself. A GPU that a
    kernel driver holds is seen by its runtime path and its kernel path, so it
    comes twice.

    [openers] counts each path's GPUs, which loads the vendor's library or reads
    the system's devices, and opens no GPU. Opening has a cost and takes the
    GPU, so a program opens only the GPU it uses.

    Raises {!Rig_cuda.exception-Refused} as {!Rig_cuda.count} does, from a
    signal handler, finaliser or [Gc.Memprof] callback that interrupts a
    call of {!Rig_cuda} on the same thread. *)
