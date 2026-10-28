(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar programs as tolk expressions, and how a graph holds nx's elements.

    {1:elements Elements}

    A value's node is a tolk tensor of the value's shape holding one element of
    {!dtype} per element of the value:
    - [float64], [float32], [float16] and [bfloat16] as tolk's types of the same
      format;
    - [float8_e4m3fn], [float8_e5m2] and [float4_e2m1fn] as their codes, the
      element's bits in the low bits of a [Uint8];
    - each integer dtype of 8 bits or more as tolk's of its width and
      signedness; [int4] as an [Int8] and [uint4] as a [Uint8] holding its
      value;
    - [bool] and [bit] as a [Uint8] holding the element's byte, true where it is
      not zero: a [bit]'s is [0] or [1];
    - [complex64] and [complex128] as [Float32] and [Float64] with one more
      axis, last, of extent [2]: the real part, then the imaginary part.

    So a sub-byte value's node holds one element per element; packing it into
    bytes is the storage's business.

    {1:numerics Numerics}

    A node computes what {!Nx_kernel.Prog} states of its kind, at the compute
    type nx's kinds state: a float narrower than [float32] computes in [float32]
    and rounds once to its dtype, an integer narrower than 32 bits in the 32-bit
    integer of its signedness. An exact kind gives the bits nx's kernels give,
    but where a float or complex kind gives a NaN, which is a NaN of no
    promised sign or payload: the integer and boolean kinds, comparisons,
    [Where], casts, bitcasts, and the float and complex kinds {!Lower_real} and
    {!Lower_complex} state exact. Every other float or complex kind holds the
    bound and the special values they state; at a float narrower than
    [float32], whose rounding absorbs [float32]'s error, it is within one rank
    of the correctly rounded result. *)

open Tolk

val dtype : Nx_array.Dtype.any -> Dtype.t
(** [dtype dt] is the type of a node's elements for a value of [dt]. *)

val dims : Nx_array.Dtype.any -> int array -> Ops.sint list
(** [dims dt s] is the shape of the node of a value of [dt] and shape [s]: [s],
    and an axis of [2] after it for a complex [dt]. *)

val parts : Nx_array.Dtype.any -> Ops.t -> Ops.t list
(** [parts dt u] is the node [u] of a value of [dt] as nodes of the value's
    shape, one per element of its node: [[u]], or a complex value's real parts,
    then its imaginary parts. *)

val of_parts : Nx_array.Dtype.any -> Ops.t list -> Ops.t
(** [of_parts dt us] is the node of a value of [dt] whose {!parts} are [us]. *)

val fill : ('v, 's) Nx_array.Dtype.t -> 'v -> Dtype.const list
(** [fill dt x] is the element [x] of [dt] as the constant of each of its
    {!parts}. Each holds the bits {!Nx_kernel.Prog.bits} gives [x], as a
    constant of a program does: a NaN keeps its sign and payload. *)

val prog : Nx_kernel.Prog.t -> int array -> Ops.t array -> Ops.t array
(** [prog p s ins] is the node of each output of [p] run at every index of the
    shape [s], its operand [i] read from the node [ins.(i)], a value of shape
    [s] and of operand [i]'s dtype. A [Coord i] is the index along axis [i] of
    [s].

    Raises [Invalid_argument] unless [ins] has one node per operand of [p]. *)

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
