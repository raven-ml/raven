(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar programs as tolk nodes, and the nodes that hold nx's elements.

    {!prog} lowers a program of {!Nx_kernel.Prog} to tolk nodes, one per output,
    and {!cast} lowers one conversion. {!Lower_real} and {!Lower_complex} lower
    the float and complex kinds a program applies.

    {1:elements Elements}

    A value's {e node} is a tolk node of the shape {!dims} gives, holding one
    element of {!dtype} per element of the value:
    - [float64], [float32], [float16] and [bfloat16] as tolk's types of the same
      format;
    - [float8_e4m3fn], [float8_e5m2] and [float4_e2m1fn] as their codes, the
      element's bits in the low bits of a [Uint8];
    - each integer dtype of 8 bits or more as tolk's of its width and
      signedness; [int4] as an [Int8] and [uint4] as a [Uint8] holding its
      value;
    - [bool] and [bit] as a [Uint8] holding the element's byte, true where it is
      not zero; a [bit]'s byte is [0] or [1];
    - [complex64] and [complex128] as [Float32] and [Float64] with one more
      axis, last, of extent [2], the real part then the imaginary part.

    A node holds each sub-byte element in a byte of its own. Only stored arrays
    pack them.

    {1:numerics Numerics}

    Each kind ({!Nx_kernel.Prog}) computes what [Prog] states for it, in its
    {e compute type}: a float narrower than [float32] computes in [float32] and
    rounds once to its dtype, an integer narrower than 32 bits in the 32-bit
    integer of its signedness, and every other dtype in itself.

    The integer and boolean kinds, the comparisons, [Where], casts and bitcasts
    are exact in {!Lower_real}'s sense, as are the float and complex kinds that
    {!Lower_real} and {!Lower_complex} state exact. Every other float or complex
    kind holds the bound and the special values those modules state for it. Such
    a kind at a dtype narrower than [float32] is within one ulp ({!Lower_real})
    of the correctly rounded result, since its [float32] error is below one ulp
    of the dtype. For [pow] this holds wherever the result is finite in that
    dtype. These rules are the elementwise part of {!Rune.section-numerics}. *)

open Tolk

(** {1:lowering Lowering} *)

val dtype : Nx_array.Dtype.any -> Dtype.t
(** [dtype dt] is the type of a node's elements for a value of [dt]. *)

val dims : Nx_array.Dtype.any -> int array -> Ops.sint list
(** [dims dt s] is [s], with an axis of extent [2] appended for a complex [dt].
*)

val parts : Nx_array.Dtype.any -> Ops.t -> Ops.t list
(** [parts dt u] is the node [u] of a value of [dt] as nodes of the value's
    shape: [[u]], or for a complex [dt] the node of its real parts then that of
    its imaginary parts. *)

val of_parts : Nx_array.Dtype.any -> Ops.t list -> Ops.t
(** [of_parts dt us] is the node of a value of [dt] whose {!parts} are [us].

    Raises [Invalid_argument] unless [us] is one node, or two for a complex
    [dt]. *)

val fill : ('v, 's) Nx_array.Dtype.t -> 'v -> Dtype.const list
(** [fill dt x] is the element [x] of [dt] as the constant of each of its
    {!parts}. Each holds the bits {!Nx_kernel.Prog.bits} gives [x], as for a
    program's constants, so a NaN keeps its sign and payload. *)

val prog : Nx_kernel.Prog.t -> int array -> Ops.t array -> Ops.t array
(** [prog p s ins] is the node of each output of [p] run at every index of the
    shape [s], its operand [i] read from the node [ins.(i)]. A [Coord i] is the
    index along axis [i] of [s]. Each [ins.(i)] must be the node of a value of
    shape [s] and of operand [i]'s dtype.

    Raises [Invalid_argument] unless [ins] has one node per operand of [p]. *)

val binary :
  Nx_kernel.Prog.binary ->
  Nx_array.Dtype.any ->
  Ops.t list ->
  Ops.t list ->
  Ops.t list
(** [binary k d xs ys] is the {!parts} of [k] applied to the elements of [d]
    whose parts are [xs] and [ys], as {!prog} applies it. It applies only
    elementwise operations, so its parts may be the scalars of a kernel. *)

val cast : src:Nx_array.Dtype.any -> Nx_array.Dtype.any -> Ops.t -> Ops.t
(** [cast ~src dt x] is the node [x] of a value of [src] stored into [dt], as
    {!Nx_kernel.Prog.Cast} stores it. *)

val bitcast :
  src:Nx_array.Dtype.any -> Nx_array.Dtype.any -> int array -> Ops.t -> Ops.t
(** [bitcast ~src dt s x] is the node [x] of a value of [src] and shape [s] read
    as [dt], as {!Nx.bitcast} reads it: of [s] at one width, with an axis of the
    ratio appended for a narrower [dt], and [s]'s last axis, of the ratio, read
    as one element for a wider one. Its bits are the bytes the host stores, each
    element's low byte first and a sub-byte element's first in the low bits. *)
