(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV ABI suites share: GPUs, kernels, words, descriptor fields, and
    descriptors drawn with the setters that made them. *)

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

val address : bits:int -> align:int -> int64 Windtrap.Gen.t
(** [address ~bits ~align] draws addresses below [2{^bits}], multiples of
    [2{^align}]. *)

(** {1:templates Templates} *)

val pp_operand : Format.formatter -> Template.operand -> unit
(** [pp_operand] prints an operand as its constructor. *)

val pp_patch : Format.formatter -> Template.patch -> unit
(** [pp_patch] prints a patch record. *)

(** {1:fields Descriptor fields} *)

val field : string -> int * int -> int
(** [field b (hi, lo)] is the field [MW(hi:lo)] of the bytes [b], the notation
    of NVIDIA's QMD headers: bits [lo] to [hi] of [b] read as one little-endian
    integer. At most 62 bits. *)

(** {1:descriptors Descriptors} *)

val dims : Qmd.dim list
(** [dims] is the six sizes of a launch. *)

val dim_name : Qmd.dim -> string
(** [dim_name d] is ["Grid X"], ["Block Z"], ... *)

val dim : Qmd.dim Windtrap.Gen.t
(** [dim] draws one of {!dims}. *)

type drawn = {
  gpu : Gpu.t;
  kernel : Cubin.kernel;
  values : int array;  (** Each value of [desc], known or not, by slot. *)
  desc : Qmd.t;
}
(** The type for descriptors drawn: a launch of [kernel] on [gpu] with the
    values [desc], each known or left to its slot. *)

val grid_slot : int
val block_slot : int
val program_slot : int
val local_slot : int
val shared_slot : int
val chain_slot : int
val bank_slot : int -> int

val release_slot : int -> int
(** The slots of [desc]'s values: the grid's sizes from [grid_slot], the block's
    from [block_slot], bank [i]'s address at [bank_slot i], and release [j]'s
    address and value from [release_slot j]. *)

val descriptor : drawn -> Template.t
(** [descriptor d] is [Qmd.make] of [d]'s launch and values. *)

val value : drawn -> int -> int64
(** [value d i] is the value of slot [i]. *)

val filled : drawn -> string
(** [filled d] is {!descriptor} filled with [d]'s values. *)

val drawn : drawn Windtrap.Gen.t
(** [drawn] draws launches of each class, of kernels whose fields fit their
    descriptor fields, and values each within the range its [.mli] states: sizes
    up to their {!Qmd.max_size}, addresses aligned and below [2{^49}] ([2{^40}]
    for chains), local memory a multiple of 16 below 1 MiB and at least the
    launch's {!Launch.local_bytes}, dynamic shared memory a multiple of 128 up
    to {!Launch.dynamic_shared}, and up to two releases, each stamped or not.
    Each value is known or a slot. *)
