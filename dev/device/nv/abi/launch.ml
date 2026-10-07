(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Repr

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type t = launch

let round_up n a = (n + a - 1) / a * a

(* The driver reserves the first 1 KiB of a block's shared memory, and 576 bytes
   of each thread's local memory. Neither NVIDIA's headers nor NVK state these
   (NVK does not run cubins); kimchi's runs of NVRTC's cubins rely on them. *)
let reserved_shared = 0x400
let reserved_local = 0x240

(* The shared memory a multiprocessor may be configured with, in KiB: the
   largest is what a launch can take. A configuration is set as its 4 KiB pages
   plus one, as NAK does (Mesa 25.2, nak/qmd.rs:338-354). *)
let shared_configs = [ 32; 64; 100 ]
let max_shared_kib = List.fold_left Int.max 0 shared_configs
let config kib = (kib * 1024 / 4096) + 1
let max_shared_config = config max_shared_kib

(* The driver's parameters at the start of bank 0, as 32-bit words: their count,
   and the words of the shared and local memory windows (64 bits each) and of
   the stack limit. The CUDA driver's layout, which NVRTC's code reads; no
   NVIDIA header states it, and kimchi's runs rely on it. *)
type parameters = { words : int; shared : int; local : int; stack : int }

let before_blackwell = { words = 12; shared = 6; local = 8; stack = 10 }
let blackwell = { words = 224; shared = 188; local = 190; stack = 223 }
let stack_limit = 0xfffdc0

(* Bank 0 when the cubin has none: the driver's parameters alone, 352 bytes. *)
let default_bank0 = { Cubin.index = 0; offset = 0; bytes = 0x160 }

let make (g : Gpu.t) (k : Cubin.kernel) =
  let layout =
    if g.compute_class = Defs.blackwell_compute_b then Defs.qmd_v5
    else if
      g.compute_class = Defs.ampere_compute_b
      || g.compute_class = Defs.ada_compute_a
    then Defs.qmd_v3
    else
      invalid_argf
        "Launch.make: compute class 0x%x, expected 0x%x, 0x%x or 0x%x"
        g.compute_class Defs.ampere_compute_b Defs.ada_compute_a
        Defs.blackwell_compute_b
  in
  (* Compared before the driver's 1 KiB is added, which a corrupt size near
     max_int would overflow. *)
  if k.shared_bytes > (max_shared_kib * 1024) - reserved_shared then
    Error
      (strf
         "the kernel declares %d bytes of shared memory, more than the %d a \
          launch leaves it beside the driver's 1 KiB"
         k.shared_bytes
         ((max_shared_kib * 1024) - reserved_shared))
  else
    let shared_bytes = round_up (reserved_shared + k.shared_bytes) 128 in
    let c = List.find (fun c -> c * 1024 >= shared_bytes) shared_configs in
    Ok
      {
        kernel = k;
        gpu = g;
        layout;
        shared_bytes;
        shared_config = config c;
        max_shared_config;
      }

let banks l =
  let banks = l.kernel.banks in
  if List.exists (fun (b : Cubin.bank) -> b.index = 0) banks then banks
  else default_bank0 :: banks

let driver_parameters l =
  let p =
    if l.gpu.compute_class = Defs.blackwell_compute_b then blackwell
    else before_blackwell
  in
  let b = Bytes.make (Int.max l.kernel.params_offset (4 * p.words)) '\000' in
  Bytes.set_int64_le b (4 * p.shared) (Int64.of_int l.gpu.shared_window);
  Bytes.set_int64_le b (4 * p.local) (Int64.of_int l.gpu.local_window);
  Bytes.set_int32_le b (4 * p.stack) (Int32.of_int stack_limit);
  Bytes.unsafe_to_string b

let local_bytes l = l.kernel.stack_bytes + reserved_local

(* A block's threads are at most 1024; fewer when their registers fill the
   register file of 65536: registers are allocated per warp of 32 threads in
   units of 256, and warps in units of 4. *)
let max_block = 1024
let register_file = 65536

let max_threads l =
  let per_warp = round_up (Int.max 1 l.kernel.registers * 32) 256 in
  Int.min max_block (register_file / per_warp / 4 * 4 * 32)
