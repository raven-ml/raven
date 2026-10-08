(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = { per_thread : int; per_tpc : int; bytes : int }

(* Rounding up and products of non-negative ints for the need [n], which raise
   where they would wrap past max_int. *)
let too_large n =
  invalid_argf "Local.make: %d bytes a thread need an allocation past max_int" n

let round_up n x a =
  if x > max_int - (a - 1) then too_large n else (x + a - 1) / a * a

let mul n a b = if b <> 0 && a > max_int / b then too_large n else a * b

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
  if n < 0 then invalid_argf "Local.make: %d bytes, expected 0 or more" n;
  let per_thread = round_up n n thread_align in
  let per_warp = round_up n (mul n per_thread warp) warp_align in
  let per_sm = mul n per_warp g.warps_per_sm in
  let per_tpc = round_up n (mul n per_sm g.sms_per_tpc) tpc_align in
  let per_gpc = mul n per_tpc g.tpcs_per_gpc in
  let bytes = round_up n (mul n per_gpc g.gpcs) bytes_align in
  { per_thread; per_tpc; bytes }
