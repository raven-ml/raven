(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hardware-level decompositions for operations not directly supported.

    Provides:

    - Transcendental decompositions: [xpow], [xsin], [xexp2], [xlog2] with
      1.0 ULP accuracy using Sleef-based polynomial approximations.
    - Counter-based PRNG: [threefry2x32].
    - Late rewrite patterns: MUL → SHL, IDIV → SHR, MOD → AND,
      MAX → WHERE, fast integer division via magic numbers.
    - Transcendental pattern factory: [get_transcendental_patterns].

    Used by {!Lowering} at pipeline steps 18-21 and by {!Symbolic} (phase 3)
    for POW folding. *)

(** {1 Transcendentals} *)

val xpow : base:Kernel.t -> exponent:Kernel.t -> Kernel.t
(** [xpow ~base ~exponent] decomposes [base ** exponent] into
    [exp2(exponent * log2(|base|))] with correct handling of negative bases,
    non-integer exponents (NaN), and [0 ** 0 = 1]. *)

val xsin : ?fast:bool -> ?switch_over:float -> Kernel.t -> Kernel.t
(** [xsin d] decomposes [sin(d)] into a 1.0 ULP polynomial approximation.
    Uses Cody-Waite reduction for small angles and Payne-Hanek reduction
    for large angles. [fast] assumes [|d| <= switch_over]. *)

val xexp2 : Kernel.t -> Kernel.t
(** [xexp2 d] decomposes [exp2(d)] into a 1.0 ULP polynomial approximation
    using the Sleef algorithm. *)

val xlog2 : Kernel.t -> Kernel.t
(** [xlog2 d] decomposes [log2(d)] into a 1.0 ULP polynomial approximation
    with denormal handling. *)

val threefry2x32 : Kernel.t -> Kernel.t -> Kernel.t
(** [threefry2x32 x key] implements the Threefry 2x32 counter-based PRNG.
    Splits uint64 [x] and [key] into uint32 halves, performs 5 rounds of
    rotation-XOR-addition, and reassembles the result as uint64. *)

(** {1 Integer division} *)

val magicgu : vmax:int -> d:int -> int * int
(** [magicgu ~vmax ~d] computes [(m, s)] such that [x // d == (x * m) >> s]
    for all [0 <= x <= vmax] and [d > 0]. Adapted from Hacker's Delight,
    Chapter 10. *)

(** {1 Long decomposition (int64 → int32 pairs)} *)

val pm_long_decomp : Kernel.t -> Kernel.t option
(** [pm_long_decomp node] decomposes int64/uint64 operations into pairs of
    int32/uint32 operations using the node tag for hi/lo tracking.
    Run conditionally when the device does not support [int64]. *)

(** {1 Float decomposition (unsupported float → supported float)} *)

type float_decomp_ctx = {
  from_dtype : Dtype.scalar;
  to_dtype : Dtype.scalar;
}
(** Context for float decomposition: convert [from_dtype] to [to_dtype]. *)

val pm_float_decomp : float_decomp_ctx -> Kernel.t -> Kernel.t option
(** [pm_float_decomp ctx node] promotes operations on unsupported float
    dtypes (fp8, bf16) to a supported dtype (typically f32).
    Run conditionally per emulated dtype pair. *)

(** {1 Late rewrite patterns} *)

(* CR: Is this the right place for late rewrite patterns? Should this live in codegen/late instead? *)

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
  disable_fast_idiv : bool;
  force_transcendental : bool;
}
(** Backend capability flags for decomposition passes. Each [has_*] flag is
    [true] iff the backend natively supports the corresponding operation.
    Unsupported operations are lowered into sequences of supported ones.

    A single flat set of supported operations consumed by both
    [get_late_rewrite_patterns] and [get_transcendental_patterns]. *)

val get_late_rewrite_patterns : supported_ops -> Kernel.t -> Kernel.t option
(** Device-specific late rewrite rules. Decomposes operations that the target
    renderer does not support directly. *)

val get_transcendental_patterns :
  supported_ops -> Kernel.t -> Kernel.t option
(** Conditionally rewrite EXP2/LOG2/SIN/SQRT into software implementations
    when the target device does not support them natively. Non-transcendental
    float dtypes (e.g. bfloat16) are cast to float32 first. *)
