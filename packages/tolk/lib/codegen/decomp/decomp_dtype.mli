(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Emulated data types.

    A target that lacks a data type computes with another one.

    - A 64-bit integer ({!Dtype.Int64}, {!Dtype.Uint64}) is two 32-bit words,
      low first, of {!Dtype.Int32} or {!Dtype.Uint32}. Its storage holds twice
      as many words, and its operations are built from 32-bit ones.
    - A narrow float ({!Dtype.Float16}, {!Dtype.Bfloat16} and the 8-bit floats)
      is stored in its own format, as the unsigned integer of its width, and
      computed with as a {!Dtype.Float32} that holds a value of the narrow
      float: a load converts it up, a store converts it down ({!f2f}), and a
      cast to it or an operation on it computes in {!Dtype.Float32} and rounds
      its result to it, once, so that the store's conversion is exact. A cast
      clamps its operand to the narrow float's range first ({!f2f_clamp}), and
      a source more precise than a {!Dtype.Float32} reaches one rounded to
      odd. A copy or a selection keeps the stored bits.

    The conversions are IEEE's: one rounding, to nearest with ties to even,
    gradual underflow, and infinities and NaNs kept where the format has them.

    A type is emulated when the target does not support it
    ({!Renderer.supported_dtypes}) or the setting {!Setting.emulated_dtypes}
    names it. *)

(** {1:floats Float conversion} *)

val f2f : ?sat:bool -> Ops.t -> Dtype.t -> Dtype.t -> Ops.t
(** [f2f ~sat v fr to_] converts between a narrow float and {!Dtype.Float32},
    [v] being the bits of a float of type [fr], as the unsigned integer of its
    width:
    - widening to {!Dtype.Float32}, it is the value [v] encodes, subnormals
      included;
    - narrowing from {!Dtype.Float32}, it is the bits, as the unsigned integer
      of [to_]'s width, of the value [v] encodes clamped ({!f2f_clamp} [~sat])
      and rounded to [to_]'s precision, to nearest with ties to even, into a
      subnormal below [to_]'s least normal.

    A NaN stays a NaN of its sign, quiet, with its payload's top bits where the
    formats have room for them; an [fnuz] format has one NaN, and no negative
    zero. [sat] defaults to [true].

    Raises [Invalid_argument] unless one of [fr] and [to_] is {!Dtype.Float32}
    and the other a narrower float. *)

val f2f_clamp : ?sat:bool -> Ops.t -> Dtype.t -> Ops.t
(** [f2f_clamp ~sat x dt] is the float [x] limited to what the narrow float [dt]
    holds. If [dt] is an 8-bit float and [sat] (default [true]), a finite
    magnitude above [dt]'s greatest finite value is that value. Otherwise a
    magnitude from [dt]'s greatest finite value plus half an ulp up is an
    infinity: rounding it to nearest even would give one. An infinity stays an
    infinity, which a format without one stores as its NaN, and a NaN a NaN. *)

val narrow : Ops.t -> Dtype.t -> Ops.t
(** [narrow x dt] is [x] cast to [dt], but for a {!Dtype.Float32} from a
    {!Dtype.Float64} or an integer of 32 or 64 bits, more precise than a
    float32: then it is [x] rounded to odd, towards zero with the last bit set
    if bits were dropped, so that a narrow float rounded to nearest from it is
    [x] rounded to nearest once. *)

(** {1:passes Passes} *)

val computes : Renderer.t -> Dtype.t list
(** [computes r] is the data types of {!Dtype.all}, in order, that [r]'s target
    computes: those it supports ({!Renderer.supported_dtypes}), and those
    {!pm_dtype_decomps} emulates where it lacks them, the 8-bit floats,
    {!Dtype.Bfloat16}, {!Dtype.Float16}, {!Dtype.Int64} and
    {!Dtype.Uint64}. *)

type ctx
(** The type for the context of {!pm_dtype_decomps}: the types a kernel uses
    that may need emulation, and the target's renderer. *)

val ctx : Renderer.t -> ctx
(** [ctx r] is the context for emulating on [r]'s target, with no type found
    yet. *)

val pm_dtype_decomps : (ctx, Ops.t) Ops.Pattern_matcher.t
(** [pm_dtype_decomps] notes the narrow floats and 64-bit integers a kernel
    uses, and at the kernel's {!Op.Sink} rewrites it to emulate those that the
    target lacks, in the promotion order of their types ({!Dtype.compare}).
    Unsigned 64-bit integers are emulated with signed ones.

    Raises [Invalid_argument] if a 64-bit integer variable must be emulated, or
    if {!Setting.emulated_dtypes} names no data type. *)
