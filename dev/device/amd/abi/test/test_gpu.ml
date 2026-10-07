(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* LLVM's names of processors, from AMDGPUUsage's processor table. *)

open Windtrap
open Device_amd_abi

let timeout = Device_amd_abi_support.timeout

let gpu target =
  {
    Gpu.target;
    gc = target;
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 4;
    compute_units = 32;
    scratch_slots = 32;
  }

let processor =
  cases ~timeout ~name:snd "a processor's name"
    [
      ((12, 0, 1), "gfx1201");
      ((11, 0, 0), "gfx1100");
      ((9, 0, 10), "gfx90a");
      ((9, 4, 2), "gfx942");
      ((9, 0, 12), "gfx90c");
      ((10, 3, 0), "gfx1030");
      ((11, 5, 1), "gfx1151");
    ]
    (fun (target, name) -> equal string name (Gpu.processor (gpu target)))

let () = exit (run "device_amd_abi.gpu" [ processor ])
