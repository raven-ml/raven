(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sources: the Linux kernel's amdgpu driver (ROCK-Kernel-Driver) for the PM4
   packet headers soc15d.h, nvd.h and kfd_pm4_headers_ai.h, the SDMA packet
   headers, and the GC register headers gc_*_offset.h and gc_*_sh_mask.h;
   GPUOpen's PAL for the PM4 packet layouts of the compute engine's firmware
   and the register spaces PM4 packets set (gfx9_plus_merged_enum.h); ROCm's
   runtime (ROCR-Runtime) for hsa.h (AQL packets), registers.h and
   amd_aql_queue.cpp (the scratch buffer descriptor and the vendor packet of
   PM4 commands); Mesa's src/amd/common/ac_sqtt.c for the program that records
   a thread trace on a compute queue; rocprofiler's counter definitions for the
   events each GPU counts. *)

module Packet = Packet
module Gpu = Gpu
module Register = Register
module Code_object = Code_object
module Scratch = Scratch
module Pm4 = Pm4
module Aql = Aql
module Sdma = Sdma
module Thread_trace = Thread_trace
module Counter = Counter
module Capability = Capability

let key : Capability.t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
