(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dtypes, layouts and arrays over device buffers.

    A {e dtype} ({!Dtype}) is what one element is: its storage format and the
    OCaml type its values read as. A {e layout} ({!Layout}) is where elements
    lie: a map from an index to an element position. A {e movement} ({!Move}) is
    a change of layout that moves no element. *)

(** {1:dtypes Dtypes} *)

(** Element formats.

    A dtype [('v, 's) t] names a storage format ['s] whose elements OCaml reads
    and writes as values of type ['v]. Operands of one format share a type, so a
    signature states which operands share one and the compiler checks every
    typed call. Where {!Bigarray} has the format, ['s] is Bigarray's element
    type, so arrays and bigarrays exchange bytes with their kinds checked by the
    type.

    A dtype's facts (its {!name}, {!bits}, {!kind} and {!float_format}) are rows
    of one table indexed by its {!code}, which the C header [nx_dtype.h] holds
    too. *)
module Dtype : sig
  (** {1:elt Storage formats}

      The second parameter of a dtype. Bigarray's element types name the formats
      Bigarray has; the others are nx's own and have no values. *)

  type float64_elt = Bigarray.float64_elt
  (** IEEE 754 binary64. *)

  type float32_elt = Bigarray.float32_elt
  (** IEEE 754 binary32. *)

  type float16_elt = Bigarray.float16_elt
  (** IEEE 754 binary16. *)

  (** bfloat16: binary32's sign, 8 exponent bits and the top 7 fraction bits. *)
  type bfloat16_elt = |

  (** OCP float8 E4M3: 4 exponent bits, 3 fraction bits, exponent bias 7, no
      infinity; [S.1111.111] is NaN. *)
  type float8_e4m3_elt = |

  (** OCP float8 E5M2: 5 exponent bits, 2 fraction bits, exponent bias 15, with
      infinities and NaNs as in IEEE 754. *)
  type float8_e5m2_elt = |

  (** OCP float4 E2M1: 2 exponent bits, 1 fraction bit, exponent bias 1, no
      infinity and no NaN. Its values are ±\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}. *)
  type float4_e2m1_elt = |

  type int64_elt = Bigarray.int64_elt
  (** Signed 64-bit integers. *)

  (** Unsigned 64-bit integers. *)
  type uint64_elt = |

  type int32_elt = Bigarray.int32_elt
  (** Signed 32-bit integers. *)

  (** Unsigned 32-bit integers. *)
  type uint32_elt = |

  type int16_signed_elt = Bigarray.int16_signed_elt
  (** Signed 16-bit integers. *)

  type int16_unsigned_elt = Bigarray.int16_unsigned_elt
  (** Unsigned 16-bit integers. *)

  type int8_signed_elt = Bigarray.int8_signed_elt
  (** Signed 8-bit integers. *)

  type int8_unsigned_elt = Bigarray.int8_unsigned_elt
  (** Unsigned 8-bit integers. *)

  (** Signed 4-bit integers, two's complement. *)
  type int4_elt = |

  (** Unsigned 4-bit integers. *)
  type uint4_elt = |

  type complex64_elt = Bigarray.complex64_elt
  (** Complex numbers of two binary64 components, real part first. *)

  type complex32_elt = Bigarray.complex32_elt
  (** Complex numbers of two binary32 components, real part first. *)

  (** Booleans, one to a byte: true iff the byte is not zero. *)
  type bool_elt = |

  (** Booleans, one to a bit. *)
  type bit_elt = |

  (** {1:dtypes Dtypes} *)

  (** The type for dtypes. An element's OCaml type depends only on its format's
      width, on every target: [float] for every float format, [int] up to 16
      bits, [int32] and [int64] for 32 and 64 bits, [Complex.t] and [bool].

      An unsigned format carries its bits in the type of its width, and the
      dtype says how to read them: {!pp_value} prints them unsigned, and the
      [Uint32] value [-1l] is 4294967295.

      Elements narrower than a byte lie LSB first within it: element [p] of a
      4-bit format is bits [4p] to [4p + 3] of the buffer, counting from bit 0
      of byte 0. *)
  type ('v, 's) t =
    | Float64 : (float, float64_elt) t
    | Float32 : (float, float32_elt) t
    | Float16 : (float, float16_elt) t
    | Bfloat16 : (float, bfloat16_elt) t
    | Float8_e4m3 : (float, float8_e4m3_elt) t
    | Float8_e5m2 : (float, float8_e5m2_elt) t
    | Float4_e2m1 : (float, float4_e2m1_elt) t
    | Int64 : (int64, int64_elt) t
    | Uint64 : (int64, uint64_elt) t
    | Int32 : (int32, int32_elt) t
    | Uint32 : (int32, uint32_elt) t
    | Int16 : (int, int16_signed_elt) t
    | Uint16 : (int, int16_unsigned_elt) t
    | Int8 : (int, int8_signed_elt) t
    | Uint8 : (int, int8_unsigned_elt) t
    | Int4 : (int, int4_elt) t
    | Uint4 : (int, uint4_elt) t
    | Complex128 : (Complex.t, complex64_elt) t
    | Complex64 : (Complex.t, complex32_elt) t
    | Bool : (bool, bool_elt) t
    | Bit : (bool, bit_elt) t

  (** The type for dtypes chosen at run time. {!equal_witness} recovers the
      static type. *)
  type any = Any : ('v, 's) t -> any

  val all : any list
  (** [all] is every dtype, in {!code} order. *)

  (** {1:facts Facts} *)

  val code : ('v, 's) t -> int
  (** [code dt] is [dt]'s code: its index in {!all} and the constant [NX_<NAME>]
      of [nx_dtype.h], as [NX_FLOAT32] for [Float32]. *)

  val bits : ('v, 's) t -> int
  (** [bits dt] is the width of one element of [dt] in bits: [1] for [Bit], [4]
      for the 4-bit formats, [8] for [Bool], up to [128] for [Complex128]. *)

  val bytes : ('v, 's) t -> int -> int
  (** [bytes dt n] is the number of bytes [n] elements of [dt] fill,
      ⌈[n · bits dt / 8]⌉.

      Raises [Invalid_argument] if [n < 0] or the number does not fit in an
      [int]. *)

  val name : ('v, 's) t -> string
  (** [name dt] is [dt]'s constructor name in lower case, as ["float32"] or
      ["float4_e2m1"]. *)

  val of_name : string -> any option
  (** [of_name s] is the dtype whose {!name} is [s], if any. *)

  val pp : Format.formatter -> ('v, 's) t -> unit
  (** [pp] formats a dtype's {!name}. *)

  (** {1:kinds Kinds} *)

  (** The type for kinds of number. ['v] is the value type of the kind's dtypes
      where the kind fixes one: integers read as [int], [int32] or [int64] by
      width. *)
  type 'v kind =
    | Float : float kind
    | Complex : Complex.t kind
    | Signed : 'v kind
    | Unsigned : 'v kind
    | Boolean : bool kind

  val kind : ('v, 's) t -> 'v kind
  (** [kind dt] is the kind of number [dt] holds. Matching it learns the value
      type: in the [Float] arm of [match kind dt with Float -> …], values of
      [dt] are [float]s. *)

  val is : 'k kind -> ('v, 's) t -> bool
  (** [is k dt] is [true] iff [kind dt] is [k]. *)

  (** {1:equality Equality} *)

  val equal : ('v, 's) t -> ('w, 'r) t -> bool
  (** [equal dt dt'] is [true] iff [dt] and [dt'] are the same dtype. *)

  val equal_witness :
    ('v, 's) t -> ('w, 'r) t -> (('v, 's) t, ('w, 'r) t) Type.eq option
  (** [equal_witness dt dt'] is [Some Equal] iff [equal dt dt']. *)

  (** {1:values Values}

      {b Stores.} Every store of a [float] into a dtype follows one rule,
      [of_float]'s here and every kernel's:

      - A finite value in range rounds once to the format's nearest value, ties
        to even. Below the least value it is a subnormal or a zero of its sign.
      - Past the largest finite value it is the infinity of its sign in
        [Float64], [Float32], [Float16] and [Bfloat16], and saturates to ±57344
        in [Float8_e5m2], ±448 in [Float8_e4m3] and ±6 in [Float4_e2m1].
      - An infinity stays one where the format has one, is NaN in [Float8_e4m3]
        and saturates to ±6 in [Float4_e2m1].
      - NaN is NaN, except in [Float4_e2m1], which has none: it stores [+0.], as
        integers do.
      - Integers truncate toward zero, saturate to their range and store NaN as
        [0], signed and unsigned alike.
      - Complex numbers store the value as their real part, rounded to their
        component's format, and a zero imaginary part; booleans store [x <> 0.].
  *)

  val zero : ('v, 's) t -> 'v
  (** [zero dt] is [dt]'s additive identity: [0], and [false] for booleans. *)

  val one : ('v, 's) t -> 'v
  (** [one dt] is [dt]'s multiplicative identity: [1], and [true] for booleans.
  *)

  val min_value : ('v, 's) t -> 'v
  (** [min_value dt] is [dt]'s least value: [neg_infinity] for a float format
      with infinities, [-448.] for [Float8_e4m3] and [-6.] for [Float4_e2m1],
      and [false] for booleans.

      Raises [Invalid_argument] if [dt] is complex. *)

  val max_value : ('v, 's) t -> 'v
  (** [max_value dt] is [dt]'s greatest value: [infinity] for a float format
      with infinities, [448.] for [Float8_e4m3] and [6.] for [Float4_e2m1], the
      value with every bit set for unsigned formats ([-1l] for [Uint32], [-1L]
      for [Uint64]), and [true] for booleans.

      Raises [Invalid_argument] if [dt] is complex. *)

  val of_float : ('v, 's) t -> float -> 'v
  (** [of_float dt x] is the value a store of [x] into [dt] holds, by the rule
      above: [of_float Float16 0.1] is [0x1.998p-4], the binary16 nearest to
      [0.1]. *)

  val pp_value : ('v, 's) t -> Format.formatter -> 'v -> unit
  (** [pp_value dt] formats a value of [dt]: a float as the shortest decimal
      that a store into [dt] reads back as the same value ([nan], [inf] and
      [-inf] for the others), an unsigned integer unsigned, a complex number as
      [re+imi]. *)

  (** {1:floats Float formats} *)

  type float_format = {
    exponent_bits : int;  (** The width of the exponent field. *)
    mantissa_bits : int;
        (** The width of the fraction field, without the implicit bit. *)
    infinities : bool;  (** Whether the format has infinities. *)
    nans : bool;  (** Whether the format has NaNs. *)
    epsilon : float;
        (** The gap between [1.] and the next value, [2{^-mantissa_bits}]. *)
    min_normal : float;  (** The least positive normal value. *)
    max_finite : float;  (** The largest finite value. *)
  }
  (** The type for the facts of a float format. *)

  val float_format : (float, 's) t -> float_format
  (** [float_format dt] is the facts of [dt]'s format. *)
end

(** {1:layouts Layouts} *)

(** Movements.

    A movement maps the indices of a result to the indices of its argument: the
    result's element at an index is the argument's element at the mapped index.
    Movements are data, so code that records or lowers them shares one
    vocabulary, and structural equality of movements is equality of their maps.
*)
module Move : sig
  type range = { start : int; count : int; step : int }
  (** The type for the elements [start + j·step], [j < count], of an axis. *)

  type window = { axis : int; size : int; step : int; dilation : int }
  (** The type for windows along [axis]: window [w]'s element [j] is the axis's
      element [w·step + j·dilation], [j < size]. *)

  (** The type for movements. *)
  type t =
    | Reshape of int array
        (** [Reshape s'] has shape [s'] and the argument's elements in the same
            C order of indices: element [k] in C order is the argument's element
            [k] in C order. *)
    | Broadcast of int array
        (** [Broadcast s'] has shape [s'], which has at least the argument's
            rank. Aligned from the right, each extent of the argument is [1] or
            [s']'s; a new axis or an extent-1 one repeats the argument's
            elements along it. *)
    | Permute of int array
        (** [Permute p] has the argument's axis [p.(i)] as its axis [i]. *)
    | Slice of range array
        (** [Slice rs] keeps the elements [rs.(i)] of each axis [i]. A negative
            step reverses the axis. *)
    | Window of window array
        (** [Window ws] has, for each window of [ws], on strictly increasing
            axes, its axis replaced by the number of windows,
            [(d - 1 - dilation·(size - 1)) / step + 1] of an axis of extent [d],
            and a trailing axis of extent [size] appended, in axis order. The
            result's element at [(…, w, …, j)] is the argument's element at
            [(…, w·step + j·dilation, …)]. *)

  val shape : t -> int array -> int array
  (** [shape m s] is the shape of the result of [m] on an argument of shape [s].

      Raises [Invalid_argument] if an extent of [s] is negative, the result has
      more than {!Layout.max_rank} axes or a number of elements that does not
      fit in an [int], or unless:
      - [Reshape s']: [s']'s extents are non-negative, their product [s]'s.
      - [Broadcast s']: [s']'s extents are non-negative; it has at least [s]'s
        rank, and each extent of [s] aligned from the right is [1] or [s']'s.
      - [Permute p]: [p] is a permutation of [s]'s axes.
      - [Slice rs]: [rs] has a range per axis, each with [step <> 0] and
        [count >= 0]; if [count > 0], [start] and [start + (count - 1)·step] lie
        in [\[0, d)] for an axis of extent [d].
      - [Window ws]: each window's axis is an axis of [s], after the previous
        window's; [size], [step] and [dilation] are at least [1] and
        [dilation·(size - 1) <= d - 1] for an axis of extent [d]. *)
end

(** Layouts.

    A layout maps an index [(i{_0}, …, i{_k-1})], [0 <= i{_j} < d{_j}], to the
    element position [offset + Σ i{_j}·s{_j}], counted in elements from a
    buffer's first byte: [d{_j}] are its extents, its {e shape}, and [s{_j}] its
    strides. An element at position [p] of a dtype of [b] bits occupies bits
    [p·b] to [p·b + b - 1] of the buffer.

    A layout is immutable and in one {e canonical form}: an axis of extent 1 has
    stride 0, and a layout with no element has offset 0 and every stride 0. Two
    layouts of one shape that map every index to the same position are then
    {!equal}.

    No OCaml array a function takes is kept, and every array a function returns
    is fresh. *)
module Layout : sig
  type t
  (** The type for layouts. *)

  val max_rank : int
  (** [max_rank] is [32], the most axes a layout has. *)

  val contiguous : int array -> t
  (** [contiguous s] is the layout of shape [s] in C order at offset 0: element
      [k] in C order is at position [k].

      Raises [Invalid_argument] if [s] has more than {!max_rank} axes, a
      negative extent, or a number of elements that does not fit in an [int]. *)

  val v : ?offset:int -> strides:int array -> int array -> t
  (** [v ~offset ~strides s] is the layout of shape [s] with [strides] and
      [offset] (defaults to [0]), in canonical form.

      Raises [Invalid_argument] as {!contiguous} does, if [strides] does not
      have [s]'s length, if [d·|s|] does not fit in an [int] for an axis of
      extent [d] and stride [s], or if a position overflows. *)

  (** {1:queries Queries} *)

  val rank : t -> int
  (** [rank l] is [l]'s number of axes. *)

  val dim : t -> int -> int
  (** [dim l i] is the extent of [l]'s axis [i].

      Raises [Invalid_argument] unless [0 <= i < rank l]. *)

  val stride : t -> int -> int
  (** [stride l i] is the stride of [l]'s axis [i].

      Raises [Invalid_argument] unless [0 <= i < rank l]. *)

  val offset : t -> int
  (** [offset l] is the position of [l]'s index [(0, …, 0)]. *)

  val numel : t -> int
  (** [numel l] is [l]'s number of indices, the product of its extents. *)

  val shape : t -> int array
  (** [shape l] is [l]'s extents. *)

  val strides : t -> int array
  (** [strides l] is [l]'s strides. *)

  val span : t -> int * int
  (** [span l] is [(lo, hi)]: every position [l] reaches lies in [\[lo, hi)],
      and [(0, 0)] if [l] has no element. *)

  val is_contiguous : t -> bool
  (** [is_contiguous l] is [true] iff element [k] of [l] in C order is at
      position [offset l + k]. *)

  val is_distinct : t -> bool
  (** [is_distinct l] is [true] if no two indices of [l] reach one position. It
      is exact for the layouts {!contiguous} reaches by [Permute], [Slice] and
      windows whose step is at least their extent [dilation·(size - 1) + 1],
      [false] for every broadcast and overlapping window, and may be [false] for
      some other strides given to {!v}: its test sorts the axes of extent above
      1 by [|stride|] and asks each stride to exceed the reach
      [Σ (d{_j} - 1)·|s{_j}|] of the smaller ones. *)

  (** {1:moving Moving} *)

  val move : Move.t -> t -> t option
  (** [move m l] is the layout of [m]'s result over [l]'s positions: its index
      [i] reaches the position [l] reaches at [m]'s map of [i]. It is [None] iff
      [m] is a [Reshape] that no strides express.

      Raises [Invalid_argument] as {!Move.shape} does on [shape l]. *)

  val coalesce : t array -> t array
  (** [coalesce ls] is layouts of one shape that reach the positions [ls] reach,
      in the same C order of indices, with their axes of extent 1 dropped and
      adjacent axes merged where every layout lays them out as one run. Each has
      at least one axis.

      Raises [Invalid_argument] unless [ls] has 1 to 4 layouts, all of one
      shape. *)

  (** {1:eq Equality} *)

  val equal : t -> t -> bool
  (** [equal l l'] is [true] iff [l] and [l'] have the same shape, strides and
      offset. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a layout's shape, strides and offset. *)
end

(** {1:arrays Arrays} *)

type ('v, 's) t
(** The type for arrays of elements of storage format ['s] read as ['v]. An
    array's layout reaches only bits of its buffer, at non-negative positions,
    and its first element lies on a multiple of its storage's alignment: its
    width for byte-wide dtypes, one component's for complex ones. *)

(** The type for arrays whose dtype is chosen at run time. {!expect} recovers
    the static type. *)
type any = Any : ('v, 's) t -> any

val v : ('v, 's) Dtype.t -> Layout.t -> Rig.Buffer.t -> ('v, 's) t
(** [v dt l b] is the array of [dt] elements laid out by [l] over [b]'s bytes.

    Raises [Invalid_argument] if [b] is dead, [l] reaches a negative position or
    a bit past [b]'s bytes, or [l] has an element and its first element's byte
    offset into [b]'s memory, or for host memory its address, is not a multiple
    of [dt]'s alignment. *)

val create :
  ?memory:Rig.Buffer.memory ->
  Rig.t ->
  ('v, 's) Dtype.t ->
  int array ->
  ('v, 's) t
(** [create d dt s] is a fresh C-contiguous array of shape [s] on [d]'s memory
    [memory] (defaults to [Device]), at offset 0, with unspecified elements. Its
    buffer holds [Dtype.bytes dt n] bytes for [n] elements; the bits of a
    sub-byte array's last byte past its last element are zero.

    Raises [Invalid_argument] as {!Layout.contiguous} does, and what
    {!Rig.Buffer.create} raises. *)

val dtype : ('v, 's) t -> ('v, 's) Dtype.t
(** [dtype a] is [a]'s dtype. *)

val layout : ('v, 's) t -> Layout.t
(** [layout a] is [a]'s layout. *)

val buffer : ('v, 's) t -> Rig.Buffer.t
(** [buffer a] is [a]'s buffer. *)

val device : ('v, 's) t -> Rig.t
(** [device a] is the device of [a]'s buffer, the device [a] lives on. *)

(** {1:moving Moving} *)

val move : Move.t -> ('v, 's) t -> ('v, 's) t option
(** [move m a] is [a]'s elements moved by [m], over [a]'s buffer, or [None]
    where {!Layout.move} is [None].

    Raises [Invalid_argument] as {!Move.shape} does. *)

val bitcast : ('w, 'r) Dtype.t -> ('v, 's) t -> ('w, 'r) t option
(** [bitcast dt a] is [a]'s bits read as elements of [dt], over [a]'s buffer.
    With [r] the ratio of the two widths:
    - equal widths keep the layout;
    - to a narrower dtype, a trailing axis of extent [r] and stride 1 is
      appended, and the other strides and the offset are multiplied by [r];
    - to a wider dtype, [a] needs a trailing axis of extent [r] and stride 1,
      and an offset and other strides that are multiples of [r]; the axis is
      removed and they are divided by [r]. An array with no element needs only
      the trailing axis of extent [r].

    It is [None] where widening's conditions fail or the result's first element
    is not on a multiple of [dt]'s alignment.

    Raises [Invalid_argument] on a narrowing of an array of rank
    {!Layout.max_rank}. *)

val expect : ('w, 'r) Dtype.t -> any -> ('w, 'r) t
(** [expect dt (Any a)] is [a] at type [dt] if its dtype is [dt].

    Raises [Invalid_argument] naming both dtypes otherwise. *)

val refused : string -> int -> any list -> 'a
(** [refused name code operands] raises [Invalid_argument] naming the kernel
    [name], the reason the code of [nx_array.h] that [nx_read] or [nx_coalesce]
    answered gives, and every operand's dtype and shape. *)

(** {1:elements Elements}

    A function that reads or writes bytes on the host claims the buffer's
    memory, waits for the device work the access must follow
    ({!Rig.Buffer.wait}), and holds the claim while it runs. It raises
    [Invalid_argument] if the buffer is dead or the host does not address its
    memory. *)

val get : ('v, 's) t -> int array -> 'v
(** [get a i] is [a]'s element at index [i].

    Raises [Invalid_argument] unless [i] has [rank] entries, each in
    [\[0, dim)]. *)

val set : ('v, 's) t -> int array -> 'v -> unit
(** [set a i x] stores [x] at [a]'s index [i]. A sub-byte element is written
    with an atomic read-modify-write of its byte, so writes to the byte's other
    elements from other domains are kept.

    Raises [Invalid_argument] as {!get} does, if [a] reaches an element twice
    ({!Layout.is_distinct}), if [a]'s memory is [Read], and if [x] is an [int]
    outside [[Dtype.min_value dt, Dtype.max_value dt]]. *)

val to_array : ('v, 's) t -> 'v array
(** [to_array a] is [a]'s elements in C order of indices, read by one C loop.
    [float] elements fill a flat [float array]; [int32], [int64] and [Complex.t]
    elements are boxed. *)

val of_array : ('v, 's) Dtype.t -> int array -> 'v array -> ('v, 's) t
(** [of_array dt s xs] is a fresh C-contiguous array of shape [s] on {!Rig.host}
    holding [xs] in C order of indices, stored by the rule of {!Dtype}.

    Raises [Invalid_argument] as {!Layout.contiguous} does, if [xs] does not
    have one value per index of [s], or if a value is an [int] outside its
    dtype's range. *)

val copy : ('v, 's) t -> ('v, 's) t
(** [copy a] is a fresh C-contiguous array on [a]'s device holding [a]'s
    elements bit for bit, NaN payloads included. *)

val to_device : Rig.t -> ('v, 's) t -> ('v, 's) t
(** [to_device d a] is [a] over a fresh buffer on [d], made by
    {!Rig.Buffer.copy} of the bytes [a]'s layout reaches between its first and
    last position; its layout is [a]'s, shifted to the copy. It copies on [a]'s
    own device too, runs no kernel and keeps strided and broadcast layouts.

    Raises what {!Rig.Buffer.copy} raises. *)

(** {1:bigarrays Bigarrays}

    {!bigarray} and {!of_bigarray} share bytes with the value they take; every
    other function makes a fresh buffer. *)

val bigarray :
  ('v, 's) Bigarray.kind ->
  ('v, 's) t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t option
(** [bigarray k a] is [Some] bigarray over [a]'s own bytes iff [a] is on
    {!Rig.host}, C-contiguous ({!Layout.is_contiguous}) and of rank at most 16:
    writes through it write [a]. From then on [a]'s memory is never held
    exclusive again ({!Rig.Buffer.bigarray}). A format Bigarray lacks is bitcast
    first to one of its width.

    Raises [Invalid_argument] if [a]'s buffer is dead, or its memory is held
    exclusive by claims that have not consumed it. *)

val of_bigarray : ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t -> ('v, 's) t
(** [of_bigarray b] is the C-contiguous array over [b]'s bytes, of [b]'s shape
    and the dtype of [b]'s kind.

    Raises [Invalid_argument] if [b]'s kind is [Char], [Int] or [Nativeint], and
    as {!v} does if [b]'s data is not aligned for its elements. *)
