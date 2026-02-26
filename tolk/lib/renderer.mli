(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** GPU kernel renderer.

    A renderer converts {!Ir.Program.t} programs to backend-specific source
    code. The abstract type {!t} encapsulates target capabilities (memory
    hierarchy, grid limits, supported operations) and a rendering function.
    Backends construct renderers via {!make}, supplying only the fields that
    differ from the defaults.

    See {!Cstyle} for C-family language backends (CUDA, Metal, OpenCL, HIP,
    Clang). *)

(** {1:types Types} *)

(** ALU operations that a backend can provide custom rendering for.

    The decomposition pass uses {!supported_ops} to decide which composite
    operations to lower; {!val-code_for_op} lists the operations a renderer
    handles natively. *)
type code_op =
  | Sqrt
  | Recip
  | Neg
  | Exp2
  | Log2
  | Sin
  | Trunc
  | And
  | Xor
  | Or
  | Add
  | Sub
  | Mul
  | Mod
  | Idiv
  | Cmpne
  | Shr
  | Shl
  | Cmplt
  | Where
  | Cmpeq
  | Fdiv
  | Max
  | Mulacc  (** Fused multiply-accumulate. *)
  | Threefry  (** Threefry 2×32 PRNG mixing function. *)

type tensor_core = {
  dims : int * int * int;  (** [(m, n, k)] matrix multiply tile dimensions. *)
  threads : int;  (** Number of threads cooperating on one tile. *)
  elements_per_thread : int * int * int;
      (** [(a, b, c)] elements each thread contributes for operands A, B, and
          accumulator C. *)
  dtype_in : Dtype.scalar;  (** Element type of the A and B input operands. *)
  dtype_out : Dtype.scalar;  (** Element type of the C accumulator operand. *)
  opts : string list;
      (** Scheduling option strings applied when this tensor core is active. *)
  swizzle :
    (string list * string list * string list)
    * (string list * string list * string list);
      (** Operand layout remapping. Each operand (A, B, C) has a triple of
          (local, upcast, reduce) dimension indices. The outer pair is (source
          swizzle, destination swizzle). *)
}
(** The type for tensor core (WMMA/MFMA) configurations.

    Describes a hardware matrix-multiply-accumulate instruction: tile geometry,
    thread mapping, dtype requirements, and the dimension swizzle needed to lay
    data out for the instruction. *)

type supported_ops = {
  has_exp2 : bool;
  has_log2 : bool;
  has_sin : bool;
  has_sqrt : bool;
  has_recip : bool;
  has_neg : bool;
  has_sub : bool;
  has_max : bool;
  has_shl : bool;
  has_shr : bool;
  has_and : bool;
  has_or : bool;
  has_cmplt : bool;
  has_cmpeq : bool;
  has_fdiv : bool;
  has_threefry : bool;
  has_mulacc : bool;
}
(** Backend capability flags consumed by the decomposition pass. Each [has_*]
    flag is [true] iff the backend natively supports the corresponding operation
    — the pass lowers unsupported operations into sequences of supported ones.

    Construct with {!supported_ops_of_code_for_op} or supply directly to
    {!make}. See {!all_supported_ops}. *)

val all_supported_ops : supported_ops
(** [all_supported_ops] marks every decomposable op as natively supported. *)

val supported_ops_of_code_for_op : code_op list -> supported_ops
(** [supported_ops_of_code_for_op ops] derives capability flags from a list of
    natively rendered operations. An operation absent from [ops] is marked
    unsupported. *)

type t
(** The type for renderers. *)

(** {1:properties Properties} *)

val name : t -> string
(** [name r] is the renderer name (e.g., ["metal"], ["cuda"]). *)

val device : t -> string
(** [device r] is the target device identifier (e.g., ["NV"], ["HIP"], ["CPU"]).
    Passed as context to codegen rewrite passes for device-specific
    transformations. *)

val has_local : t -> bool
(** [has_local r] is [true] iff [r] supports local thread IDs. *)

val has_threads : t -> bool
(** [has_threads r] is [true] iff [r] supports host-side threading instead of
    GPU grid dimensions. *)

val has_shared : t -> bool
(** [has_shared r] is [true] iff [r] supports shared memory. *)

val global_max : t -> int list option
(** [global_max r] is the maximum global grid dimensions [[x; y; z]], or [None]
    when unconstrained. The list has exactly three elements when present. *)

val local_max : t -> int list option
(** [local_max r] is the maximum local workgroup dimensions [[x; y; z]], or
    [None] when unconstrained. The list has exactly three elements when present.
*)

val shared_max : t -> int
(** [shared_max r] is the maximum shared memory size in bytes.

    - [0] when shared memory is unsupported ({!has_shared} is [false]).
    - For GPU backends, a conservative default (e.g., 32KB for OpenCL, 48KB for
      CUDA). Actual limits may vary by device. *)

val tensor_cores : t -> tensor_core list
(** [tensor_cores r] is the list of {!type:tensor_core} configurations supported
    by [r]. *)

(** {1:rendering Rendering} *)

val render : t -> ?name:string -> Ir.Program.t -> string
(** [render r ~name program] converts [program] to backend-specific source code.
    [name] is the kernel function name, defaults to ["kernel"]. *)

(** {1:load_store Load/Store policy} *)

val load_store_widths : t -> Dtype.t -> int list
(** [load_store_widths r dtype] is the preferred vector widths for load/store
    coalescing of [dtype], ordered from widest to narrowest. The list must
    include [1] (scalar fallback).

    The devectorizer uses this list to split wide accesses into the largest
    widths the backend supports. The default (via {!make}) is [[1]] (scalar
    only). *)

val code_for_op : t -> code_op list
(** [code_for_op r] is the list of ALU operations that [r] provides custom
    rendering for. *)

val supported_ops : t -> supported_ops
(** [supported_ops r] is the backend capability flags for the decomposition
    pass, derived from {!val-code_for_op} unless explicitly overridden via
    {!make}. *)

(** {1:construction Construction} *)

val make :
  ?tensor_cores:tensor_core list ->
  ?load_store_widths:(Dtype.t -> int list) ->
  ?has_threads:bool ->
  ?global_max:int list option ->
  ?local_max:int list option ->
  ?code_for_op:code_op list ->
  ?supported_ops:supported_ops ->
  name:string ->
  device:string ->
  has_local:bool ->
  has_shared:bool ->
  shared_max:int ->
  render:(?name:string -> Ir.Program.t -> string) ->
  unit ->
  t
(** [make ~name ~device ~has_local ~has_shared ~shared_max ~render ()] is a
    renderer with the given capabilities.

    Optional parameters and their defaults:
    - [tensor_cores]: [[]] (none).
    - [load_store_widths]: [fun _ -> [1]] (scalar only).
    - [has_threads]: [false].
    - [global_max]: [Some [0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF]].
    - [local_max]: [Some [0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF]].
    - [code_for_op]: [[]] (no custom ops).
    - [supported_ops]: derived from [code_for_op]; if [code_for_op] is [[]],
      defaults to {!all_supported_ops}.

    For use by backend implementations. *)
