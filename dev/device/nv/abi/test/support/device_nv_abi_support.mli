(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV ABI suites share: GPUs, kernels, words, descriptor fields, and
    descriptors drawn with the setters that made them. *)

open Device_nv_abi

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
  ?compute_class:int -> ?shared_window:int -> ?local_window:int -> unit -> Gpu.t
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

(** {1:fields Descriptor fields} *)

val field : string -> int * int -> int
(** [field b (hi, lo)] is the field [MW(hi:lo)] of the bytes [b], the notation
    of NVIDIA's QMD headers: bits [lo] to [hi] of [b] read as one little-endian
    integer. At most 62 bits. *)

(** {1:descriptors Descriptors} *)

(** The type for the setters of {!Qmd}, with their arguments. *)
type op =
  | Set_dim of Qmd.dim * int
  | Patch_dim of Qmd.dim * int64
  | Set_program of int64
  | Set_bank of int * int64
  | Set_local_memory of int64
  | Release of Packet.scope * int64 * int64
  | Release_stamp of Packet.scope * int64 * int64
  | Chain of int64

val pp_op : Format.formatter -> op -> unit
(** [pp_op] prints an operation as the call it makes. *)

val dims : Qmd.dim list
(** [dims] is the six sizes of a launch. *)

val dim_name : Qmd.dim -> string
(** [dim_name d] is ["Grid X"], ["Block Z"], ... *)

val dim : Qmd.dim Windtrap.Gen.t
(** [dim] draws one of {!dims}. *)

val scope_name : Packet.scope -> string
(** [scope_name s] is ["Agent"] or ["System"]. *)

val scope : Packet.scope Windtrap.Gen.t
(** [scope] draws a scope. *)

val apply : op -> int64 Qmd.t -> int64 Qmd.t
(** [apply op q] is [q] with [op]; a release of a descriptor whose releases are
    both taken leaves it as it was. *)

type drawn = { gpu : Gpu.t; kernel : Cubin.kernel; ops : op list }
(** The type for descriptors drawn: a launch of [kernel] on [gpu], then [ops].
*)

val descriptor : drawn -> int64 Qmd.t
(** [descriptor d] is [Qmd.make] of [d]'s launch with [d.ops] applied in order.
*)

val drawn : drawn Windtrap.Gen.t
(** [drawn] draws launches of each class, of kernels whose fields fit their
    descriptor fields, and setters each within the range its [.mli] states:
    sizes up to their {!Qmd.max_size}, addresses aligned and below [2{^49}]
    ([2{^40}] for releases and chains), local memory a multiple of 16 below 1
    MiB and at least the launch's {!Launch.local_bytes}. No size is both set and
    patched: that case is test_qmd's. *)
