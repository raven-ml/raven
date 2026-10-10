(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** N-dimensional arrays as values on device sets.

    A value of type [('v, 's, 'd) t] is elements of one dtype, read as ['v] and
    stored as ['s], with a shape and a placement over the device set ['d].
    Nothing changes a value.

    A device set is a module: [module Gpu = (val Nx.devices [ d0; d1 ])] mints a
    brand [Gpu.d], and values of two sets never meet in typed code. *)

(** {1:values Values}

    A value is a shape, a dtype and the devices its elements lie on. *)

type (!'v, !'s, +!'d) t
(** The type for arrays of ['s]-format elements read as ['v], on the device set
    ['d]. *)

type ('v, 's) dtype = ('v, 's) Nx_array.Dtype.t
(** The type for dtypes whose elements OCaml reads as ['v] and which store them
    as ['s]. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is [x]'s extents, a fresh array. *)

val dtype : ('v, 's, 'd) t -> ('v, 's) dtype
(** [dtype x] is [x]'s element format. *)

(** {1:dtypes Dtypes}

    A dtype says how a value stores its elements and how OCaml reads them:
    [float32] stores 32-bit floats and reads them as [float], [int8] stores
    8-bit integers and reads them as [int]. The operands of an operation share a
    dtype by their types; {!cast} changes it. There is one value per constructor
    of {!Dtype.t}, named in lowercase. *)

module Dtype = Nx_array.Dtype
(** Dtypes: their constructors, element types and properties. *)

val float64 : (float, Dtype.float64_elt) dtype
val float32 : (float, Dtype.float32_elt) dtype
val float16 : (float, Dtype.float16_elt) dtype
val bfloat16 : (float, Dtype.bfloat16_elt) dtype
val float8_e4m3fn : (float, Dtype.float8_e4m3fn_elt) dtype
val float8_e5m2 : (float, Dtype.float8_e5m2_elt) dtype
val float4_e2m1fn : (float, Dtype.float4_e2m1fn_elt) dtype
val int64 : (int64, Dtype.int64_elt) dtype
val uint64 : (int64, Dtype.uint64_elt) dtype
val int32 : (int32, Dtype.int32_elt) dtype
val uint32 : (int32, Dtype.uint32_elt) dtype
val int16 : (int, Dtype.int16_signed_elt) dtype
val uint16 : (int, Dtype.int16_unsigned_elt) dtype
val int8 : (int, Dtype.int8_signed_elt) dtype
val uint8 : (int, Dtype.int8_unsigned_elt) dtype
val int4 : (int, Dtype.int4_elt) dtype
val uint4 : (int, Dtype.uint4_elt) dtype
val complex128 : (Complex.t, Dtype.complex64_elt) dtype
val complex64 : (Complex.t, Dtype.complex32_elt) dtype
val bool : (bool, Dtype.bool_elt) dtype
val bit : (bool, Dtype.bit_elt) dtype

(** Each dtype value [dt] names the type of its values, [dt_t]: a
    ['d float32_t] is a float32 value on the set ['d]. *)

type 'd float64_t = (float, Dtype.float64_elt, 'd) t
type 'd float32_t = (float, Dtype.float32_elt, 'd) t
type 'd float16_t = (float, Dtype.float16_elt, 'd) t
type 'd bfloat16_t = (float, Dtype.bfloat16_elt, 'd) t
type 'd float8_e4m3fn_t = (float, Dtype.float8_e4m3fn_elt, 'd) t
type 'd float8_e5m2_t = (float, Dtype.float8_e5m2_elt, 'd) t
type 'd float4_e2m1fn_t = (float, Dtype.float4_e2m1fn_elt, 'd) t
type 'd int64_t = (int64, Dtype.int64_elt, 'd) t
type 'd uint64_t = (int64, Dtype.uint64_elt, 'd) t
type 'd int32_t = (int32, Dtype.int32_elt, 'd) t
type 'd uint32_t = (int32, Dtype.uint32_elt, 'd) t
type 'd int16_t = (int, Dtype.int16_signed_elt, 'd) t
type 'd uint16_t = (int, Dtype.int16_unsigned_elt, 'd) t
type 'd int8_t = (int, Dtype.int8_signed_elt, 'd) t
type 'd uint8_t = (int, Dtype.int8_unsigned_elt, 'd) t
type 'd int4_t = (int, Dtype.int4_elt, 'd) t
type 'd uint4_t = (int, Dtype.uint4_elt, 'd) t
type 'd complex128_t = (Complex.t, Dtype.complex64_elt, 'd) t
type 'd complex64_t = (Complex.t, Dtype.complex32_elt, 'd) t
type 'd bool_t = (bool, Dtype.bool_elt, 'd) t
type 'd bit_t = (bool, Dtype.bit_elt, 'd) t

(** {1:creation Creation}

    A value made from a dtype, a shape and numbers, or by operations from such
    values alone, is a value of every set: its type is polymorphic in ['d]. It
    is a formula and holds no bytes. Each operation that needs its elements
    computes it on that operation's set, {!place} computes it at a placement,
    and a read computes it on the host. Its {!placement} is [None]. An operation
    that reads such a value keeps the elements it computed, once per placement,
    for as long as the value lives; the values it was computed from keep none
    for it. {!place} gives a value memory of its own. {!zeros_like} and {!copy}
    make a value from another, where it lies, and of every set from a value of
    every set. *)

val zeros : ('v, 's) dtype -> int array -> ('v, 's, 'd) t
(** [zeros dt s] is the value of shape [s] whose every element is zero, of every
    set.

    Raises [Invalid_argument] if an extent is negative. *)

val scalar : ('v, 's) dtype -> 'v -> ('v, 's, 'd) t
(** [scalar dt v] is the 0-d value [v], of every set.

    Raises [Invalid_argument] if [v] is an [int] outside [dt]'s range. *)

val zeros_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [zeros_like x] is zeros of [x]'s dtype and shape, where [x] lies. *)

val donate : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [donate x] is [x], given up by its owner: a handle that one operation reads.
    [x] stays usable until that operation is called. When it is called, [x] and
    the handle die: a later operation on either raises [Invalid_argument] naming
    the consumer, and its shape, dtype and placement still answer.

    A movement that maps the elements one to one ({!reshape}, {!flatten},
    {!squeeze}, {!unsqueeze}, {!transpose}, {!moveaxis}, {!swapaxes}, {!flip},
    {!rearrange}) spends its handle and passes a new one to its result, whether
    it makes a view or a copy; the deaths wait for the final consumer. Any other
    operation consumes it. Two handles of one donor reaching one operation raise
    naming [Nx.donate]: the same handle twice, two donations, or one chain read
    twice.

    The consumer may write its result into the donated memory when it reads the
    operand only at the result's own index (an elementwise operand), the memory
    is C-contiguous with the result's shape and dtype, and the memory is not
    shared. Memory is shared once a value outside the donor's handle chain has
    been made over it (a view, a borrow by {!place}, a {!Repr} crossing), and
    stays shared. Otherwise the consumer computes into fresh memory, and a value
    that shares the memory keeps it. Its result is the same either way.

    A traced value and a value of every set are returned unchanged: nothing
    dies, and passing such a value twice raises nothing.

    Raises [Invalid_argument] if [x] is dead. *)

val copy : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [copy x] is [x] stored afresh: equal elements in new memory. *)

(** {1:shapes Shapes, broadcasting and movements}

    Axes count from [0], and a negative axis from the end: [-1] is the last.
    Shapes broadcast from their last axes: aligned there, each pair of extents
    is equal or one of them is [1], and missing leading axes count as [1].

    A movement rearranges elements without computing on them. Each says whether
    it makes a {e view}, a value over its operand's memory that costs no copy.
    That is a cost, never part of the result's meaning: a value never changes,
    so a view and a copy read the same. *)

val ndim : ('v, 's, 'd) t -> int
(** [ndim x] is [x]'s number of axes. *)

val dim : int -> ('v, 's, 'd) t -> int
(** [dim a x] is the extent of [x]'s axis [a].

    Raises [Invalid_argument] if [a] is not an axis of [x]. *)

val numel : ('v, 's, 'd) t -> int
(** [numel x] is [x]'s number of elements: the product of its extents, [1] for a
    0-d value. *)

val nbytes : ('v, 's, 'd) t -> int
(** [nbytes x] is the bytes [x]'s elements fill when stored contiguously,
    [(numel x * b + 7) / 8] for a dtype of [b] bits. *)

val reshape : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [reshape s x] is [x]'s elements in C order at shape [s]. One extent of [s]
    may be [-1]; it is inferred from the others. A view where strides express
    it, a copy otherwise.

    Raises [Invalid_argument] if the element counts differ, an extent is below
    [-1], or two are [-1]. *)

val broadcast_to : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [broadcast_to s x] is [x] stretched to shape [s]: an axis of extent [1], or
    one [x] lacks, repeats [x]'s elements along it. A view.

    Raises [Invalid_argument] unless [x]'s shape broadcasts with [s] to [s]. *)

val broadcast_shapes : int array list -> int array
(** [broadcast_shapes ss] is the shape that [ss] broadcast to, [[||]] for [[]].

    Raises [Invalid_argument] if an extent is negative or two of them do not
    broadcast. *)

val broadcast_arrays : ('v, 's, 'd) t list -> ('v, 's, 'd) t list
(** [broadcast_arrays xs] is each of [xs] stretched to {!broadcast_shapes} of
    their shapes, in order. Views.

    Raises [Invalid_argument] if two of their shapes do not broadcast. *)

val squeeze : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [squeeze ~axes x] is [x] without the axes [axes]; without [axes], without
    every axis of extent [1]. A view.

    Raises [Invalid_argument] if an axis of [axes] is not [x]'s, repeats, or has
    an extent other than [1]. *)

val unsqueeze : axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [unsqueeze ~axes x] is [x] with an axis of extent [1] at each position of
    [axes], positions of the result: [unsqueeze ~axes:[ 0; -1 ]] of a [[3]]
    value is [[1; 3; 1]]. A view.

    Raises [Invalid_argument] if a position is not an axis of the result or
    repeats. *)

val flatten : ?start_dim:int -> ?end_dim:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [flatten ~start_dim ~end_dim x] is [x] with its axes [start_dim] to
    [end_dim], both included, merged into one. [start_dim] defaults to [0] and
    [end_dim] to [-1]; a 0-d [x] flattens as the [[1]] value it holds. A view
    where strides express it, a copy otherwise.

    Raises [Invalid_argument] if either is not an axis of [x] or [start_dim]
    comes after [end_dim]. *)

val transpose : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [transpose ~axes x] has [x]'s axis [List.nth axes i] as its axis [i];
    without [axes], [x]'s axes in reverse order. A view.

    Raises [Invalid_argument] unless [axes] lists each of [x]'s axes once. *)

val moveaxis : int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [moveaxis a b x] is [x] with its axis [a] moved to position [b], the other
    axes in their order. A view.

    Raises [Invalid_argument] if [a] or [b] is not an axis of [x]. *)

val swapaxes : int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [swapaxes a b x] is [x] with its axes [a] and [b] exchanged. A view.

    Raises [Invalid_argument] if [a] or [b] is not an axis of [x]. *)

val flip : ?axes:int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [flip ~axes x] is [x] with the order of elements along each axis of [axes]
    reversed; without [axes], along every axis. A view.

    Raises [Invalid_argument] if an axis of [axes] is not [x]'s or repeats. *)

val sliding_window :
  ?axis:int -> window:int -> ?step:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sliding_window ~axis ~window ~step x] is [x]'s windows of [window]
    consecutive elements along [axis] (default [-1]), one every [step] (default
    [1]). Axis [axis] becomes the number of windows, [(d - window) / step + 1]
    for an extent [d], and an axis of extent [window] is appended: window [w]'s
    element [j] is [x]'s element [w * step + j] along [axis]. A view, whose
    windows share elements where [step < window].

    Raises [Invalid_argument] if [axis] is not [x]'s, [window] or [step] is
    below [1], or [window] exceeds [d]. *)

val split : axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t list
(** [split ~axis n x] is [x] cut along [axis] into [n] runs of consecutive
    elements, in order. With [d] the extent, the first [d mod n] runs have
    [d / n + 1] elements and the others [d / n]. Views.

    Raises [Invalid_argument] if [axis] is not [x]'s or [n < 1]. *)

val tile : int array -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tile reps x] is [x] repeated [reps.(i)] times, end to end, along its axis
    [i]. With more entries in [reps] than [x] has axes, [x] first gains leading
    axes of extent [1]. A view where strides express it, a copy otherwise.

    Raises [Invalid_argument] if [reps] has fewer entries than [x] has axes or a
    negative one. *)

val repeat : ?axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [repeat ~axis n x] is [x] with each element repeated [n] times in place
    along [axis]: [repeat ~axis:0 2] of [[a; b]] is [[a; a; b; b]]. Without
    [axis], [flatten x] repeated. A view where strides express it, a copy
    otherwise.

    Raises [Invalid_argument] if [axis] is not an axis of [x] or [n < 0]. *)

(** {1:patterns Axis patterns}

    A pattern states a layout in words. Names are words separated by spaces,
    parentheses group axes merged into one, [1] is an axis of extent [1], and
    [...] stands for the axes no name covers. One grammar serves {!rearrange},
    with one operand, and einsum and contract, with two:

    {v
    pattern  ::= operands "->" layout [ "|" name { name } ]
    operands ::= layout [ "," layout ]
    layout   ::= { axis }
    axis     ::= name | "(" name { name } ")" | "1" | "..."
    name     ::= a letter, then letters, digits and "_"
    v}

    {[
    let split_heads = Nx.Pattern.v "b t (h d) -> b h t d"
    let merge_heads = Nx.Pattern.inverse split_heads
    let q = Nx.rearrange ~sizes:[ ("h", heads) ] split_heads y
    ]} *)

(** Axis patterns. *)
module Pattern : sig
  type t
  (** The type for axis patterns, parsed and checked. *)

  val v : string -> t
  (** [v s] is the pattern [s]. Each call parses [s]; a pattern bound at a
      module's top level is parsed once, as the module starts.

      With one operand, every name appears once on each side; a group on the
      left splits an axis and on the right merges axes; [1] drops a unit axis on
      the left and adds one on the right; [...] is the same axes on both sides.

      With two, a name in both operands and the result is a batch axis, in one
      operand and the result a free axis, and the names after [|] are summed:
      each is in both operands and not in the result. Every operand name is in
      the result or summed. A group in an operand splits that operand's axis,
      and an extent a group leaves unknown comes from the other operand where
      the name appears there. [...] stands for the same leading axes in all three
      layouts, kept as batch axes; [1] drops a unit axis of an operand or adds
      one to the result.

      Raises [Invalid_argument] naming [s] if it is not in the grammar, a name
      repeats within a layout, [...] appears twice in a layout, or a name breaks
      the rules above. A string in NumPy's einsum notation, such as
      ["bik,bkj->bij"], raises with the pattern it means:
      ["b i k, b k j -> b i j | k"]. *)

  val inverse : t -> t
  (** [inverse p] is the one-operand pattern that undoes [p]: its sides
      exchanged.

      Raises [Invalid_argument] if [p] has two operands. *)
end

val rearrange :
  ?sizes:(string * int) list -> Pattern.t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rearrange ~sizes p x] is [x] split, permuted and merged as [p] says: the
    result's element at an index is [x]'s element where each name takes the same
    position. A name alone on the left takes its axis's extent, and [sizes]
    gives others; a group on the left may hold one name whose extent neither
    gives, the quotient of its axis by the others. A view where strides express
    it, a copy otherwise.

    Raises [Invalid_argument] if [p] has two operands, [p]'s left side names
    more axes than [x] has or, without [...], other than [x]'s rank, a [1] on
    the left meets an extent other than [1], a group's extents do not divide or
    multiply to its axis, a group has two unknown extents, a size disagrees with
    the extent [x] gives its name, a name in [sizes] is not in [p] or is there
    twice, or a size is negative. *)

(** {1:arith Arithmetic}

    Elementwise operations compute where their operands lie and give a new
    value. Binary operations broadcast: aligned at their last axes, two extents
    are equal or one of them is [1], which stretches. Integers wrap. The float
    formats narrower than 32 bits compute at [float32] and round once.

    Each raises [Invalid_argument] naming itself if the shapes do not broadcast,
    or if its operation does not take the operands' dtype: what each takes is
    stated with it. The host computes every dtype stated; on a set whose kernels
    do not compute one, as the GPU libraries complex numbers today, the function
    raises naming itself. *)

val add : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [add a b] is the elementwise sum. Every dtype but booleans. *)

val sub : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sub a b] is [a - b], as {!add}. *)

val mul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mul a b] is the elementwise product, as {!add}. *)

val div : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [div a b] is [a / b]: the IEEE 754 quotient on floats and complex numbers;
    on integers the quotient truncated toward zero, [0] for a divisor of [0],
    and a signed dtype's least value for that value by [-1]. Every dtype but
    booleans. *)

val mod_ : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mod_ a b] is the remainder of [a / b], of [a]'s sign: [fmod] on floats, and
    [a] for a divisor of [0] on integers, so that
    [a = add (mul b (div a b)) (mod_ a b)] on integers. Floats and integers. *)

val pow : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pow a b] is [a] to the power [b]. Floats and integers. *)

val fma : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [fma a b c] is [a * b + c] rounded once. Floats and integers. *)

val maximum : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [maximum a b] is the greater element by the dtype's order ({!less}). On
    floats it is the IEEE 754 maximum: a NaN gives NaN, and [-0.] orders below
    [+0.]. Every dtype. *)

val minimum : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [minimum a b] is the lesser element, as {!maximum}. *)

val neg : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [neg x] is [-x]. Every dtype but booleans. *)

val recip : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [recip x] is [1 / x]; on integers, [x] for [1] and [-1] and [0] otherwise.
    Every dtype but booleans. *)

val abs : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [abs x] is [x]'s magnitude. Floats and integers. *)

val sign : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sign x] is [-1], [0] or [1] by [x]'s sign, and NaN for a NaN. Floats and
    integers. *)

(** {2:transcendental Powers, exponentials and trigonometry}

    Floats only. *)

val sqrt : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sqrt x] is the square root, NaN below [-0.]. *)

val exp : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [exp x] is [e{^x}]. *)

val exp2 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [exp2 x] is [2{^x}], exact at integers whose power the dtype holds. *)

val expm1 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [expm1 x] is [e{^x} - 1], accurate near [0]. *)

val log : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log x] is the natural logarithm: [-inf] at [0.], NaN below. *)

val log2 : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log2 x] is the base-2 logarithm, exact at powers of two. *)

val log1p : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [log1p x] is [log (1 + x)], accurate near [0]. *)

val sin : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sin x] is the sine of [x] radians. *)

val cos : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cos x] is the cosine of [x] radians. *)

val tan : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tan x] is the tangent of [x] radians. *)

val asin : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [asin x] is the arcsine, in \[[-π/2], [π/2]\]. *)

val acos : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [acos x] is the arccosine, in \[[0], [π]\]. *)

val atan : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan x] is the arctangent, in \[[-π/2], [π/2]\]. *)

val atan2 : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan2 y x] is the angle of the point [(x, y)], in \][-π], [π]\]. *)

val sinh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sinh x] is the hyperbolic sine. *)

val cosh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cosh x] is the hyperbolic cosine. *)

val tanh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tanh x] is the hyperbolic tangent. *)

val erf : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erf x] is the error function [2/√π ∫₀ˣ e{^-t²} dt]. *)

(** {2:rounding Rounding}

    Floats and integers; the identity on integers. *)

val floor : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [floor x] rounds toward negative infinity. *)

val ceil : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [ceil x] rounds toward positive infinity. *)

val round : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [round x] rounds to the nearest integer, halves away from zero. *)

val trunc : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [trunc x] rounds toward zero. *)

(** {2:compare Comparisons and bits}

    Every dtype has one order: [false < true]; integers by value, unsigned
    dtypes unsigned; floats by value, [-0.] below [+0.]; complex numbers by real
    part, then imaginary part. A comparison is [true] where it holds; it
    compares floats by value, so [-0.] equals [+0.], and it is [false] where an
    element is a NaN, or a complex number with a NaN part, but for {!not_equal},
    which is [true] there. Comparisons take every dtype. *)

val equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [equal a b] is [a = b]. *)

val not_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [not_equal a b] is [a <> b]. *)

val less : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [less a b] is [a < b]. *)

val less_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [less_equal a b] is [a <= b]. *)

val greater : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [greater a b] is [less b a]. *)

val greater_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [greater_equal a b] is [less_equal b a]. *)

val bitwise_and : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_and a b] is the bitwise and of integers, the logical and of
    booleans. Integers and booleans. *)

val bitwise_or : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_or a b] is the bitwise or, as {!bitwise_and}. *)

val bitwise_xor : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_xor a b] is the bitwise exclusive or, as {!bitwise_and}. *)

val where : 'd bool_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [where c x y] is [x]'s element where [c] is [true] and [y]'s elsewhere.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

(** {2:reductions Reductions and scans}

    A reduction folds [x]'s elements along [axes], every axis by default, a
    negative one counted from the end, and drops them, or keeps each with extent
    [1] where [keepdims]. Its order of association depends on the shape alone. A
    float result that a NaN reaches is the first NaN in C order. The float
    formats narrower than 32 bits accumulate at [float32] and round once.

    Each raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated, or a dtype it does not take. *)

val sum : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sum ~axes x] adds the elements from [+0]: a sum of [-0.] terms is [+0.],
    and a sum of none is [+0.]. Integers wrap. Every dtype but booleans. *)

val prod : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [prod ~axes x] multiplies the elements from [1], as {!sum}. Floats and
    integers. *)

val max : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [max ~axes x] is the greatest element, as {!maximum} orders them: [-0.]
    below [+0.], any on booleans. Floats, integers and booleans.

    Raises [Invalid_argument] where an axis it reduces is empty. *)

val min : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [min ~axes x] is the least element, as {!max}; all on booleans. *)

val mean : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mean ~axes x] is the {!sum} divided by the number of elements, rounded
    once: NaN for none. Floats and complex numbers. *)

val cumsum : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumsum ~axis x] is, at each index, the sum of [x]'s elements along [axis]
    up to that index, inclusive; without [axis], of its elements in C order up
    to that one, in [x]'s shape. Every dtype but booleans.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank or a
    dtype it does not take. *)

val cumprod : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumprod ~axis x] is the running product, as {!cumsum}. Floats and integers.
*)

val cummax : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummax ~axis x] is the running {!max}, as {!cumsum}. Floats, integers and
    booleans. *)

val cummin : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummin ~axis x] is the running {!min}, as {!cummax}. *)

(** {2:conversion Conversion} *)

val cast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [cast dt x] is [x]'s elements stored in [dt]; [x] itself where [dt] is [x]'s
    dtype.

    A float rounds once to the nearest value of a float format, ties to even.
    Past the format's largest finite value it is an infinity in [float64],
    [float32], [float16] and [bfloat16], and saturates in the formats of a byte
    or less. NaN stays NaN, but in [float4_e2m1fn], where it is [+0.]. Into an
    integer, a float truncates toward zero and saturates to the range, NaN
    giving [0]; an integer is kept modulo the width; a boolean is [0] or [1].
    Into a boolean, any element is [true] where it is not zero, NaN included. An
    integer into a float rounds once from its exact value. Into a complex dtype
    these rules give the real part, and the imaginary part is [0.]; a complex
    number stores part by part into a complex dtype, as [true] into a boolean if
    a part is not zero, and by its real part into any other dtype. *)

val bitcast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [bitcast dt x] is [x]'s bits read as elements of [dt], NaN payloads
    included: [x] itself where [dt] is [x]'s dtype. At one width the shape is
    [x]'s. A [dt] [k] times narrower appends an axis of [k], its elements the
    pieces of [x]'s element, in the host's byte order; one [k] times wider reads
    [x]'s last axis, of [k] elements, as one element, and drops it.

    Raises [Invalid_argument] where [dt] is wider and [x]'s last axis does not
    have [k] elements, and where [dt] is narrower and [x] has the greatest rank
    already. *)

(** {1:devices Device sets and placement}

    A value lies on a device set, a module minted by {!devices} whose brand ['d]
    keeps its values apart from other sets'. Within its set a value has a
    placement: whole on one or every device, or cut into windows across them.
    {!place} moves a value between sets, and between placements of one. The host
    is a set like any other, {!Host}. *)

type host
(** The brand of {!Host}. *)

type +'d devices
(** The type for device sets of brand ['d]: distinct devices, and the kernels
    that compute on them. *)

val rigs : 'd devices -> Rig.t list
(** [rigs s] is [s]'s devices in order. *)

(** Named axes over a set's devices. *)
module Mesh : sig
  type +'d t
  (** The type for named axes over a set of brand ['d]. *)

  val v : 'd devices -> (string * int) list -> 'd t
  (** [v s axes] lays [s]'s devices in row-major order over the named [axes].

      Raises [Invalid_argument] unless the extents are positive and multiply to
      the number of [s]'s devices and the names are distinct. *)
end

(** Where a value's elements lie over its set. *)
module Placement : sig
  (* CR: Expose [device : 'd devices -> Rig.t -> 'd t], checking membership.
     No public constructor selects one member of a multi-device set. A new
     singleton set changes the brand, preventing a Check's condition and data
     from occupying different devices of one set. Recovering this placement
     through Nx_array and Repr needs storage solely to name a placement that
     Devices.one already caches. *)
  type +'d t
  (** The type for placements over a set of brand ['d]. *)

  val on : 'd devices -> 'd t
  (** [on s] holds the whole value on every device of [s]. *)

  val split : axis:int -> 'd devices -> 'd t
  (** [split ~axis s] cuts [axis] into equal windows, one per device of [s], in
      order.

      Raises [Invalid_argument] if [axis] is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
  (** [mesh m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
      [names], major first; a mesh axis no cut names holds copies.
      [split ~axis s] is [mesh (Mesh.v s [ ("x", n) ]) [ (axis, [ "x" ]) ]].

      Raises [Invalid_argument] for a name [m] lacks, an axis or a name in two
      cuts, or an axis that is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val devices : 'd t -> 'd devices
  (** [devices p] is [p]'s set. *)

  val equal : 'd t -> 'd t -> bool
  (** [equal p q] is [true] iff every device holds the same window of a value at
      [p] and at [q]. *)

  val pp : Format.formatter -> 'd t -> unit
  (** [pp] formats a placement on one device as the device's name, others as
      [on [CUDA:0; CUDA:1]], [split ~axis:0 [CUDA:0; CUDA:1]] or a mesh's
      extents and cuts. *)
end

(** The type for device sets as modules: a brand and its placements. *)
module type Devices = sig
  type d
  (** The set's brand. *)

  val v : d devices
  (** [v] is the set. *)

  val on : d Placement.t
  (** [on] is [Placement.on v]. *)

  val split : axis:int -> d Placement.t
  (** [split ~axis] is [Placement.split ~axis v]. *)
end

module Host : Devices with type d = host
(** The process's host, {!Rig.host}, computed by nx.cpu. *)

val devices : ?kernels:(module Nx_kernel.S) -> Rig.t list -> (module Devices)
(** [devices ~kernels ds] mints a new set over [ds], in their order, with a
    brand no other set has, computed eagerly by [kernels]. Without [kernels], a
    set whose every device runs on the host ({!Rig.runs_on_host}) is computed by
    nx.cpu, and any other set by none.

    Raises [Invalid_argument] if [ds] is empty or repeats a device, or if
    [kernels] does not compute on one of [ds]. *)

val place : 'e Placement.t -> ('v, 's, 'd) t -> ('v, 's, 'e) t
(** [place p x] is [x]'s elements at [p]. Each device of [p] holds its window
    over [x]'s own memory where [x] already lies there, or where [x]'s memory is
    the host's and that device shares host memory ({!Rig.shares_host_memory});
    otherwise it holds a copy of its window, allocating at most the window's
    bytes. A value of every set is computed at [p]. Across sets it changes the
    brand; within one, the arrangement.

    Raises [Invalid_argument] if [p]'s cuts do not divide [x]'s shape, and
    {!Rig.Lost} for a lost device. *)

val placement : ('v, 's, 'd) t -> 'd Placement.t option
(** [placement x] is where [x]'s bytes are, or [None] for a value of every set,
    which has none. *)

(** {1:random Random numbers} *)

(** Random numbers from one Threefry-2x32 generator.

    A key is the generator's whole state: every draw is a function of its key,
    dtype and shape, the same on every set. Fresh draws come from fresh keys, by
    {!split} or {!fold_in}. A key is spent once it is used: its own draws, its
    {!split} keys, its {!fold_in} keys and the keys of a scope rooted at it all
    come from one stream of blocks under it, so using a key two of these ways
    can repeat bits. A scope lets one key stand for a region: a sampler called
    without [~key] takes the scope's next key.

    A draw is a value of every set when its key is: a formula that computes on
    the set of each operation that reads it. *)
module Rng : sig
  type +'d key = private 'd int32_t
  (** The type for keys, of shape [[|2|]], and batches of keys, of shape
      [[|…; 2|]]: two 32-bit words per key. Arithmetic on a key gives an array,
      which no sampler takes. *)

  (** {1:keys Keys} *)

  val key : int -> 'd key
  (** [key seed] is the key of [seed], a value of every set. Equal seeds give
      equal keys. *)

  val of_tensor : 'd int32_t -> 'd key
  (** [of_tensor t] is the key, or batch of keys, whose words are [t], for words
      that were saved: the inverse of {!to_tensor}.

      Raises [Invalid_argument] unless [t]'s last axis has extent [2]. *)

  val to_tensor : 'd key -> 'd int32_t
  (** [to_tensor k] is [k]'s words: the inverse of {!of_tensor}, and what the
      coercion of [k] to an array is. *)

  val place : 'e Placement.t -> 'd key -> 'e key
  (** [place p k] is [k] at [p], as {!Nx.place}; a draw from it computes at [p].
  *)

  val split : ?n:int -> 'd key -> 'd key array
  (** [split ~n k] is [n] keys derived from [k] ([n] defaults to [2]),
      independent of each other. They are [k]'s blocks [0] to [n - 1], so a key
      split is spent ({!Rng}).

      Raises [Invalid_argument] if [n < 1] or [k] is a batch. *)

  val split_batch : n:int -> 'd key -> 'd key
  (** [split_batch ~n k] is [split ~n k] as one batch of shape [[|n; 2|]]: row
      [i] holds [(split ~n k).(i)]'s words.

      Raises [Invalid_argument] if [n < 1] or [k] is a batch. *)

  val fold_in : 'd key -> int -> 'd key
  (** [fold_in k i] is the key of [k] indexed by [i]: distinct [i] give
      independent keys. A batch of keys gives a batch. *)

  val fold_in_tensor : 'd key -> 'd int32_t -> 'd key
  (** [fold_in_tensor k i] is [fold_in k] of the indices held in [i], for an
      index known only as data. Its shape is [k]'s batch broadcast with [i]'s,
      then [2]; where [i] holds [n] it is [fold_in k n].

      Raises [Invalid_argument] if the shapes do not broadcast. *)

  (** {1:samplers Samplers}

      A sampler draws from [~key], or without it from the scope's next key. A
      key with batch axes [b] draws [b] followed by the draw's shape, one draw
      per key, each the draw that key alone gives. Float draws compute at
      float64 for float64 and at float32 for the other floats, and round once to
      their dtype.

      A sampler's parameters are arrays, and the draw has their broadcast
      shape. Each parameter is checked against its domain: an element outside
      raises [Invalid_argument] naming the sampler, the parameter, the element's
      index and its value, as in
      [Nx.Rng.bernoulli: p at [3] is 1.5, not in [0, 1]]. The check runs as the
      sampler is called, computing the parameter if it is a formula; under an
      interpretation it is a [Check] operation, which raises
      where the interpretation computes it. *)

  val bits : ?key:'d key -> int array -> 'd int32_t
  (** [bits shape] is uniformly random 32-bit words: word [j] in C order is word
      [j mod 2] of the generator's block [j / 2].

      Raises [Invalid_argument] if an extent is negative. *)

  val uniform :
    ?key:'d key -> (float, 's) dtype -> int array -> (float, 's, 'd) t
  (** [uniform dt shape] is draws from [\[0, 1)]: multiples of [2{^-p}], each
      equally likely, [p] the largest such that every multiple of [2{^-p}] in
      [\[0, 1)] is a value of [dt]: [dt]'s significand width, from float64's
      53 down to float8_e5m2's 3; float4_e2m1fn's values below 1 are 0 and
      0.5, so its [p] is 1.

      Raises [Invalid_argument] if an extent is negative. *)

  val normal :
    ?key:'d key -> (float, 's) dtype -> int array -> (float, 's, 'd) t
  (** [normal dt shape] is standard normal draws, by the Box-Muller transform of
      two uniform draws.

      Raises [Invalid_argument] if an extent is negative. *)

  val exponential :
    ?key:'d key -> (float, 's) dtype -> int array -> (float, 's, 'd) t
  (** [exponential dt shape] is exponential draws of rate 1, by inverting the
      distribution at [1 - u] for a uniform [u]: finite, and never negative.

      Raises [Invalid_argument] if an extent is negative. *)

  val randint :
    ?key:'d key ->
    ?low:int ->
    high:int ->
    int array ->
    'd int32_t
  (** [randint ~low ~high shape] is integers drawn uniformly from
      [\[low, high)] ([low] defaults to [0]): a 64-bit draw times [high - low],
      shifted down by 64 bits, whose biased low part, at most one draw in
      [2{^32}], takes a second draw (Lemire's multiply-shift).

      Raises [Invalid_argument] if [low >= high], a bound is outside int32, or
      an extent is negative. *)

  val bernoulli :
    ?key:'d key -> (float, 's, 'd) t -> 'd bool_t
  (** [bernoulli p] is [true] with probability [p] rounded up to a multiple of
      [2{^-53}], elementwise: 53 random bits, read as an integer, below
      [p 2{^53}].

      Raises [Invalid_argument] if an element of [p] is outside [[0, 1]]. *)

  val gamma : ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [gamma a] is gamma draws of concentration [a] and unit rate, elementwise:
      Marsaglia and Tsang's rejection, below a concentration of 1 through
      [Gamma(a) = Gamma(a + 1) U{^1/a}]. Divide by a rate for the two-parameter
      family.

      It is not exact: a draw takes the first of eight rounds that accepts, each
      accepting more than 98% of proposals, and the one element in about
      [10{^14}] that no round accepts is the distribution's mean.

      Raises [Invalid_argument] if an element of [a] is outside [(0, inf)]. *)

  val beta :
    ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [beta a b] is beta draws on [[0, 1]] of concentrations [a] and [b],
      elementwise over their broadcast shape: [G(a) / (G(a) + G(b))] for two
      independent {!gamma} draws, formed from their logarithms so that two draws
      that underflow keep their ratio. It inherits {!gamma}'s rounds.

      Raises [Invalid_argument] if the shapes do not broadcast or an element of
      [a] or [b] is outside [(0, inf)]. *)

  val von_mises : ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [von_mises k] is von Mises draws of mean direction 0 and concentration
      [k], elementwise, as angles in [[-π, π]]: uniform on the circle at
      [k = 0], near a normal of variance [1 / k] for a large [k].

      It is not exact: Best and Fisher's rejection from a wrapped Cauchy
      envelope runs twenty rounds, and the one element in about [2·10{^9}] that
      no round accepts is a draw from the envelope. Add a mean direction,
      wrapped into [\[-π, π\]], for the general family.

      Raises [Invalid_argument] if an element of [k] is outside [\[0, inf)]. *)

  val poisson :
    ?key:'d key -> (float, 's, 'd) t -> 'd int32_t
  (** [poisson rate] is Poisson counts of [rate], elementwise; a rate of 0 gives
      0. Below 10 a count inverts the distribution with one uniform draw, with
      no fallback. From 10 up it is Hörmann's transformed rejection over
      sixteen rounds, and the one element in about [5·10{^9}] that no round
      accepts takes its last proposal, a count near the rate clamped to at
      least 0. A count past int32's largest saturates there. A float32 rate
      places its proposals exactly up to about [10{^5}]; beyond, float32's
      spacing near the rate exceeds a hundredth of a count, so proposals drift
      off their counts and the distribution is biased: give a float64 rate.

      Raises [Invalid_argument] if an element of [rate] is outside
      [\[0, 2{^31})]. *)

  val binomial :
    ?key:'d key ->
    'd int32_t ->
    (float, 's, 'd) t ->
    'd int32_t
  (** [binomial n p] is the number of successes in [n] trials that each succeed
      with probability [p], elementwise over their broadcast shape: 0 at
      [p = 0], [n] at [p = 1]. Where the mean of the rarer outcome is below 10 a
      count inverts the distribution with one uniform draw, with no fallback;
      from 10 up it is Hörmann's transformed rejection over eighteen rounds,
      and the one element in about [4·10{^9}] that no round accepts takes its
      last proposal, clamped into [\[0, n\]]. A float32 [p] places its
      proposals exactly up to a mean of about [10{^5}]; beyond, float32's
      spacing near the mean exceeds a hundredth of a count, so proposals drift
      off their counts and the distribution is biased: give a float64 [p].

      Raises [Invalid_argument] if the shapes do not broadcast, an element of
      [n] is negative, or one of [p] is outside [[0, 1]]. *)

  (** {1:scope Scope} *)

  val with_key : 'd key -> (unit -> 'a) -> 'a
  (** [with_key k f] runs [f] in a scope rooted at [k]: the samplers called in
      [f] without [~key] take successive keys of [k], [fold_in k 0],
      [fold_in k 1], …. Scopes nest, the inner replacing the outer. A scope is
      per fiber and per domain.

      A keyless draw is a value of every set, so its key must be one: raises
      [Invalid_argument] if [k] has bytes ({!place}d, or made by {!of_tensor}
      from an array that has). *)

  val next_key : unit -> 'd key
  (** [next_key ()] is the scope's next key, a value of every set. Outside every
      scope, it is the next key of the domain's own, seeded from the system's
      entropy. *)
end

(** {1:errors Errors}

    Every misuse raises [Invalid_argument] whose message is the function the
    user called, a colon, the reason and, where they matter, the operands as
    [dtype shape on set], as in
    [Nx.place: axis 0 of extent 6 does not split evenly over 4]. A lost device
    raises {!Rig.Lost}, and a device's memory that runs out
    {!Rig.Out_of_memory}. Host memory that runs out raises OCaml's
    [Out_of_memory]. *)

(** {1:repr Arrays}

    The crossing to the array layer, for libraries that read or make bytes:
    formats, C bindings, compilers. Layouts and buffers are visible here and
    nowhere else in this module. It shares memory, copying nothing: a value and
    the arrays crossing with it read the same bytes, so a caller who writes them
    breaks "nothing changes a value". {!copy} gives a value memory of its own.
*)

module Repr : sig
  val of_array : 'd devices -> ('v, 's) Nx_array.t -> ('v, 's, 'd) t
  (** [of_array s a] is the value whose bytes are [a]'s. The value reads [a]'s
      memory from then on: a write to it through any handle leaves what values
      over it read unspecified.

      Raises [Invalid_argument] naming [Nx.Repr.of_array] unless [a] lies on a
      device of [s]. *)

  val array : ('v, 's, 'd) t -> ('v, 's) Nx_array.t option
  (** [array x] is [Some a] iff [x] has bytes on one device, [a] its array, and
      [None] for a value on several devices or of every set. [a] may be strided
      or offset. It is for reading: writing through it changes [x]. *)

  val of_shards : 'd Placement.t -> ('v, 's) Nx_array.t array -> ('v, 's, 'd) t
  (** [of_shards p arrays] is the value whose bytes are [arrays], one per device
      of [p] in order, each that device's window.

      Raises [Invalid_argument] naming [Nx.Repr.of_shards] unless there is one
      array per device of [p], each on its device, all of one shape, of a rank
      that has every axis [p] cuts. *)

  val shards : ('v, 's, 'd) t -> ('v, 's) Nx_array.t array option
  (** [shards x] is [Some arrays], one per device of [x]'s placement, in order:
      [[| a |]] for a value on one device; [None] for a value of every set. The
      arrays are for reading, as {!array}'s. *)
end

(**/**)

(** Operations as data, for interpreters such as rune's.

    An operation goes to the innermost live interpretation that reaches it.
    Every interpretation reaches the operations on its traced values. An
    [Extent] interpretation also reaches every operation the calling fiber
    applies inside its extent, on its domain. Innermost is by start order. An
    interpretation does not reach the operations its own running rule applies,
    and that rule applying an operation to the interpretation's own traced
    values raises. An operation no interpretation reaches computes now on its
    operands' set's kernels, and raises if the set has none. Each refusal raises
    [Invalid_argument] naming [by] and the interpretation.

    A value of every set reaches a rule computed where the operation reads it,
    beside an operand with a placement. An operation over values of every set
    alone reaches it with them as they are, formulas with no placement. *)
module Prim : sig
  type ('v, 's, 'd) nx := ('v, 's, 'd) t

  type ('v, 's, 'd) form = {
    dtype : ('v, 's) dtype;
    layout : Nx_array.Layout.t;
    placement : 'd Placement.t option;
  }
  (** A value without its bytes. A value cut over devices has the C-contiguous
      layout of the whole. *)

  type 'd any = Any : ('v, 's, 'd) nx -> 'd any

  type 'd load =
    | Plain : ('v, 's, 'd) nx -> 'd load
        (** How a loop reads an operand: through its layout. *)

  (** What a loop's reduction makes of one output of its program. *)
  type ('d, _) reduction =
    | Monoid :
        Nx_kernel.Spec.monoid * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx) reduction
        (** [Monoid (m, k, dt)] folds output [k] by [m] in that output's dtype
            and rounds once to [dt]. *)
    | Moments :
        int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx * ('v, 's, 'd) nx) reduction
        (** The population mean of output [k], then its variance. *)
    | Arg :
        Nx_kernel.Spec.extreme * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx * (int64, Dtype.int64_elt, 'd) nx) reduction
        (** The extreme of output [k], then its first position in C order of the
            reduced indices. *)

  type ('d, 'r) reductions =
    | [] : ('d, unit) reductions
    | ( :: ) :
        ('d, 'a) reduction * ('d, 'r) reductions
        -> ('d, 'a * 'r) reductions
        (** A reduction's reductions, its results in order. *)

  type ('d, 'r) outs =
    | [] : ('d, unit) outs
    | ( :: ) : ('v, 's) dtype * ('d, 'r) outs -> ('d, ('v, 's, 'd) nx * 'r) outs
        (** The dtypes of a map's results. *)

  (** The operations. A loop's loads have exactly its iteration shape: no
      operation broadcasts or promotes. *)
  type 'r t =
    | Map : {
        layout : Nx_array.Layout.t;
        prog : Nx_kernel.Prog.t;
        outs : ('d, 'r) outs;
        loads : 'd load array;
      }
        -> 'r t
        (** [prog] at every index of [layout]'s shape, reading load [i] as its
            operand [i]; result [k] is its output [k], laid out as [layout],
            which is C-contiguous. A one-result map is ['v * unit]. With no
            loads, a creation. *)
    | Reduce : {
        layout : Nx_array.Layout.t;
        axes : int array;
        prog : Nx_kernel.Prog.t;
        reductions : ('d, 'r) reductions;
        loads : 'd load array;
      }
        -> 'r t
        (** [prog] at every index of [layout]'s shape, as a map's, each
            reduction folding its output along [axes], strictly increasing, as
            {!Nx_kernel.Spec.reduce} says. Results drop [axes] and are
            C-contiguous. [Max], [Min] and [Arg] of no term are refused. *)
    | Scan : {
        layout : Nx_array.Layout.t;
        axis : int;
        prog : Nx_kernel.Prog.t;
        reduction : ('d, 'r) reduction;
        loads : 'd load array;
      }
        -> 'r t
        (** The reduction of [prog]'s output over the indices along [axis] up to
            each index's, inclusive, of [layout]'s shape. [Moments] is refused.
        *)
    | Copy : ('v, 's, 'd) nx -> ('v, 's, 'd) nx t
        (** The value stored afresh, C-contiguous. *)
    | Move : Nx_array.Move.t * ('v, 's, 'd) nx -> ('v, 's, 'd) nx t
    | Bitcast : ('w, 'r) dtype * ('v, 's, 'd) nx -> ('w, 'r, 'd) nx t
        (** The bits read in another dtype: one width keeps the shape, a
            narrower one appends an axis of the ratio, a wider one consumes a
            trailing axis of it. *)
    | Place : 'e Placement.t * ('v, 's, 'd) nx -> ('v, 's, 'e) nx t
    | Check : {
        ok : (bool, Dtype.bool_elt, 'd) nx;
        data : 'd any list;
        fail : int array -> 'd any list -> exn;
      }
        -> unit t
        (** Raises [fail i data_i] at the first index [i], in C order, where
            [ok] is [false], [data_i] each of [data] at [i]. *)

  type operands = Operands : 'd any list -> operands

  val name : 'r t -> string
  (** [name op] is [op]'s constructor, as ["Map"]. *)

  val pp : Format.formatter -> 'r t -> unit
  (** [pp] formats an operation: its name, its programs as expressions over its
      operands, and each operand's dtype, shape and placement. *)

  val operands : 'r t -> operands
  (** [operands op] is [op]'s operands in order: a map's loads, [Check]'s [ok]
      then its data, the one operand of the others. *)

  val map : ('v 's 'd. ('v, 's, 'd) nx -> ('v, 's, 'd) nx) -> 'r t -> 'r t
  (** [map m op] is [op] with each operand [x] replaced by [m x]. *)

  val form : ('v, 's, 'd) nx -> ('v, 's, 'd) form
  (** [form x] is [x] without its bytes. *)

  val results :
    by:string ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) nx) ->
    'r t ->
    'r
  (** [results ~by m op] is [op]'s result, its value at position [k] made by
      [m k f], [f] the form eager execution gives it; [()] for [Check].

      Raises [Invalid_argument] naming [by], before [m] is called, where [op]'s
      operands break its rule. *)

  (** {1:interpretations Interpretations} *)

  type interpretation
  (** The type for interpretations. *)

  type reach =
    | Values  (** Reaches the operations on its traced values. *)
    | Extent
        (** Also reaches every operation its starting fiber applies inside its
            extent, on its domain. *)

  type ('v, 's, +'d) payload = ..
  (** What an interpretation keeps in its traced values. A payload holds values,
      never a function of ['d]. *)

  val interpret :
    name:string ->
    reach ->
    ('r. interpretation -> by:string -> 'r t -> 'r) ->
    (interpretation -> 'a) ->
    'a
  (** [interpret ~name reach rule f] is [f i], [i] a new interpretation that
      gives the operations it reaches the meaning [rule i ~by op], live until
      [f] returns or raises. [name] names it in messages, as ["Rune.grad"]. *)

  val traced :
    interpretation ->
    ('v, 's, 'd) form ->
    ('v, 's, 'd) payload ->
    ('v, 's, 'd) nx
  (** [traced i f p] is a value of form [f] that [i] owns, keeping [p]. Its
      form, payload and owner answer after [i] returns; any operation on it then
      raises. *)

  val payload : interpretation -> ('v, 's, 'd) nx -> ('v, 's, 'd) payload option
  (** [payload i x] is [Some p] iff [i] owns [x], [p] what it keeps. *)

  val owner : ('v, 's, 'd) nx -> interpretation option
  (** [owner x] is the interpretation that owns [x], if [x] is traced. *)

  val later : interpretation -> interpretation -> bool
  (** [later a b] is [true] iff [a] started after [b].

      Raises [Invalid_argument] if they started on two domains. *)

  val eval : by:string -> 'r t -> 'r
  (** [eval ~by op] is [op]'s meaning under the rule, as the vocabulary's
      functions apply it. *)

  val expand : interpretation -> by:string -> 'r t -> 'r option
  (** [expand i ~by op] is [Some r], [r] [op] as core operations applied by
      {!eval} with [i] not running, so that they reach [i] again; [None] for an
      operation that has no expansion. A map of several nodes expands into one
      map per node. *)
end

(**/**)
