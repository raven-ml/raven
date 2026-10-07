(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { per_thread : int; per_tpc : int; bytes : int }

let round_up n a = (n + a - 1) / a * a

(* The alignments of NVK's chain (Mesa 25.2, nvk_device.c:55-95): a warp's share
   to 512 bytes, a TPC's to 32 KiB (SET_SHADER_LOCAL_MEMORY_NON_THROTTLED
   requires it) and the allocation to 128 KiB. A thread's share is rounded to 32
   bytes first. *)
let thread_align = 32
let warp_align = 0x200
let tpc_align = 0x8000
let bytes_align = 0x20000
let warp = 32

let make (g : Gpu.t) n =
  if n < 0 then
    invalid_arg (Printf.sprintf "Local.make: %d bytes, expected 0 or more" n);
  let per_thread = round_up n thread_align in
  let per_warp = round_up (per_thread * warp) warp_align in
  let per_tpc =
    round_up (per_warp * g.warps_per_sm * g.sms_per_tpc) tpc_align
  in
  let bytes = round_up (per_tpc * g.tpcs_per_gpc * g.gpcs) bytes_align in
  { per_thread; per_tpc; bytes }
