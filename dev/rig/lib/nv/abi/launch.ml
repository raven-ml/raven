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

(* Bank 0 when the cubin has none: the driver's parameters alone, 352 bytes. *)
let default_bank0 = { Cubin.index = 0; offset = 0; bytes = 0x160 }

(* What a kernel may use, as CUDA states it for compute capabilities 8.0 to 12.0
   (CUDA C++ Programming Guide, "Technical Specifications per Compute
   Capability"), and the banks a descriptor names, 0 to 7 (clc7c0qmd.h,
   clcec0qmd.h: CONSTANT_BUFFER_VALID(i)). Every descriptor field holds them. *)
let max_registers = 255
let max_bank_bytes = 0x10000
let descriptor_banks = 8

(* The shared memory a launch leaves the kernel beside the driver's 1 KiB. *)
let max_kernel_shared = (max_shared_kib * 1024) - reserved_shared

(* What of [k] no launch takes, if anything: a check that allocates nothing when
   [k] fits. *)
let bank_refusal (b : Cubin.bank) =
  if b.index < 0 || b.index >= descriptor_banks then
    Some
      (strf "the kernel reads constant bank %d, expected 0 to %d" b.index
         (descriptor_banks - 1))
  else if b.bytes > max_bank_bytes then
    Some
      (strf "the kernel's constant bank %d is %d bytes, expected at most %d"
         b.index b.bytes max_bank_bytes)
  else None

let rec banks_refusal = function
  | [] -> None
  | b :: bs -> (
      match bank_refusal b with None -> banks_refusal bs | refused -> refused)

(* Compared before the driver's 1 KiB is added, which a corrupt size near
   max_int would overflow. *)
let refusal (k : Cubin.kernel) =
  if k.shared_bytes > max_kernel_shared then
    Some
      (strf
         "the kernel declares %d bytes of shared memory, expected at most %d \
          (the driver keeps 1 KiB)"
         k.shared_bytes max_kernel_shared)
  else if k.registers > max_registers then
    Some
      (strf "the kernel uses %d registers per thread, expected at most %d"
         k.registers max_registers)
  else banks_refusal k.banks

(* The smallest configuration that holds [bytes]. *)
let rec config_for bytes = function
  | [] -> max_shared_config
  | c :: cs -> if c * 1024 >= bytes then config c else config_for bytes cs

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
  let max_sass_version = (1 lsl layout.sass_version.bits) - 1 in
  if g.sass_version < 0 || g.sass_version > max_sass_version then
    invalid_argf "Launch.make: SASS version %d, expected 0 to %d" g.sass_version
      max_sass_version;
  match refusal k with
  | Some why -> Error why
  | None ->
      let shared_bytes = round_up (reserved_shared + k.shared_bytes) 128 in
      Ok
        {
          kernel = k;
          gpu = g;
          layout;
          shared_bytes;
          shared_config = config_for shared_bytes shared_configs;
          max_shared_config;
        }

let banks l =
  let banks = l.kernel.banks in
  if List.exists (fun (b : Cubin.bank) -> b.index = 0) banks then banks
  else default_bank0 :: banks

let local_bytes l = l.kernel.stack_bytes + reserved_local
let dynamic_shared l = (max_shared_kib * 1024) - l.shared_bytes

(* A block's threads are at most 1024; fewer when their registers fill the
   register file of 65536: registers are allocated per warp of 32 threads in
   units of 256, and warps in units of 4. *)
let max_block = 1024
let register_file = 65536

let max_threads l =
  let per_warp = round_up (Int.max 1 l.kernel.registers * 32) 256 in
  Int.min max_block (register_file / per_warp / 4 * 4 * 32)
