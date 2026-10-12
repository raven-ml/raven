(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV ABI suites share: GPUs, kernels, words and patches. *)

open Rig_nv_abi

(** {1:gpus GPUs and kernels} *)

val ampere : int
(** [ampere] is AMPERE_COMPUTE_B, [0xc7c0]. *)

val ada : int
(** [ada] is ADA_COMPUTE_A, [0xc9c0]. *)

val blackwell : int
(** [blackwell] is BLACKWELL_COMPUTE_B, [0xcec0]. *)

val classes : int list
(** [classes] is [[ampere; ada; blackwell]], the classes {!Gpu.t} names. *)

val class_name : int -> string
(** [class_name c] is ["ampere"], ["ada"], ["blackwell"] or ["0x..."]. *)

val gpu :
  ?compute_class:int ->
  ?sass_version:int ->
  ?shared_window:int ->
  ?local_window:int ->
  unit ->
  Gpu.t
(** [gpu ()] is an RTX 5000 Ada: class {!ada}, SASS 8.9, 11 GPCs of 6 TPCs of 2
    SMs of 48 warps, windows at [0x7294_0000_0000] and [0x7293_0000_0000], and a
    [local] that does nothing, unless said otherwise. *)

val kernel :
  ?code_bytes:int ->
  ?registers:int ->
  ?shared_bytes:int ->
  ?stack_bytes:int ->
  ?params_offset:int ->
  ?banks:Cubin.bank list ->
  unit ->
  Cubin.kernel
(** [kernel ()] is a kernel of 0x100 bytes of code at 0x80, 32 registers, 0x20
    bytes of stack, and no shared memory, parameters or banks, unless said
    otherwise. *)

val launch : Gpu.t -> Cubin.kernel -> Launch.t
(** [launch g k] is [Launch.make g k]. Raises [Failure] on [Error]. *)

val compute_class : int Windtrap.Gen.t
(** [compute_class] draws one of {!classes}. *)

val bank : Cubin.bank Windtrap.testable
(** [bank] is the witness of constant banks. *)

val pp_kernel : Format.formatter -> Cubin.kernel -> unit
(** [pp_kernel] prints a kernel record. *)

(** {1:words Words} *)

val words : string -> int list
(** [words s] is the 32-bit little-endian words of [s], as unsigned integers. *)

val u64 : int64 Windtrap.Gen.t
(** [u64] draws 64-bit integers, often at the edges of their halves and of the
    type. *)

val le64 : int64 -> string
(** [le64 n] is the 8 little-endian bytes of [n]. *)

val round_up : int -> int -> int
(** [round_up n a] is the least multiple of [a] at or above [n], [a > 0]. *)

(** {1:templates Templates} *)

val pp_patch : Format.formatter -> Template.patch -> unit
(** [pp_patch] prints a patch record. *)
