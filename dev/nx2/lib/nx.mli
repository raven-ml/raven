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

(** The type for values whose dtype is hidden, such as a file's named values.
    Match on [P] for the value; {!unpack} also fixes its dtype. *)
type 'd packed = P : ('v, 's, 'd) t -> 'd packed

val unpack : ('v, 's) dtype -> 'd packed -> ('v, 's, 'd) t
(** [unpack dt p] is the value in [p] if its dtype is [dt].

    Raises [Invalid_argument] naming both dtypes otherwise. *)

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

val ones : ('v, 's) dtype -> int array -> ('v, 's, 'd) t
(** [ones dt s] is the value of shape [s] whose every element is one ([true]
    for booleans), of every set.

    Raises [Invalid_argument] if an extent is negative. *)

val full : ('v, 's) dtype -> int array -> 'v -> ('v, 's, 'd) t
(** [full dt s v] is the value of shape [s] whose every element is [v], a float
    stored as {!Dtype.of_float} says, of every set.

    Raises [Invalid_argument] if an extent is negative or [v] is an [int] outside
    [dt]'s range. *)

val scalar : ('v, 's) dtype -> 'v -> ('v, 's, 'd) t
(** [scalar dt v] is the 0-d value [v], of every set.

    Raises [Invalid_argument] if [v] is an [int] outside [dt]'s range. *)

val arange : ('v, 's) dtype -> int -> int -> int -> ('v, 's, 'd) t
(** [arange dt start stop step] is the 1-d value [start], [start + step], …
    before [stop], of every set: empty where [stop] is not past [start] in
    [step]'s direction. Each value is exact, then stored in [dt] as {!cast}
    stores an [int64].

    Raises [Invalid_argument] if [step = 0], or a value is outside [dt]'s range
    for an integer dtype, or other than [0] and [1] for a boolean one. *)

val arange_f : (float, 's) dtype -> float -> float -> float -> (float, 's, 'd) t
(** [arange_f dt start stop step] is the 1-d value of the [⌈(stop - start) /
    step⌉] values [start + i step], each computed at float64 with one rounding,
    then stored in [dt], of every set; empty where that count is not positive.

    Raises [Invalid_argument] if [step = 0.] or the count is not finite. *)

val linspace :
  ('v, 's) dtype -> ?endpoint:bool -> float -> float -> int -> ('v, 's, 'd) t
(** [linspace dt ~endpoint start stop n] is [n] values evenly spaced from
    [start] to [stop], [stop] included where [endpoint] (default [true]), of
    every set. They are computed at float64 and stored in [dt]: the first is
    [start] and, with [endpoint] and [n >= 2], the last is [stop], each as [dt]
    stores it, and for finite [start] and [stop] every value lies between them.

    Raises [Invalid_argument] if [n < 0]. *)

val logspace :
  (float, 's) dtype ->
  ?endpoint:bool ->
  ?base:float ->
  float ->
  float ->
  int ->
  (float, 's, 'd) t
(** [logspace dt ~endpoint ~base start stop n] is [base] (default [10.]) to the
    powers [linspace float64 ~endpoint start stop n], each power computed at
    float64 and stored once in [dt], of every set: [logspace float32 0. 2. 3]
    is [1], [10] and [100].

    Raises [Invalid_argument] if [n < 0]. *)

val eye : ?m:int -> ?k:int -> ('v, 's) dtype -> int -> ('v, 's, 'd) t
(** [eye ~m ~k dt n] is the [n × m] value ([m] defaults to [n]) whose element at
    row [i] and column [j] is one where [j - i = k] (default [0]) and zero
    elsewhere, of every set: [k > 0] is a diagonal above the main one.

    Raises [Invalid_argument] if [n] or [m] is negative. *)

(** {2:like From a reference}

    These make a value of their argument's dtype where it lies: beside a value
    of every set, a value of every set. Its elements are never read. *)

val zeros_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [zeros_like x] is zeros of [x]'s dtype and shape, at [x]'s placement. *)

val ones_like : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [ones_like x] is {!ones} of [x]'s dtype and shape, at [x]'s placement. *)

val full_like : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [full_like x v] is {!full} of [x]'s dtype and shape, at [x]'s placement.

    Raises [Invalid_argument] as {!full} does. *)

val scalar_like : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [scalar_like x v] is {!scalar} of [x]'s dtype, whole on each device of
    [x]'s placement: a 0-d value has no axis to cut.

    Raises [Invalid_argument] as {!scalar} does. *)

(** {2:moving Donation and copies} *)

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

(** {2:ocaml From OCaml values}

    A value made from OCaml data lies on the host, the set {!Host}. *)

type host
(** The brand of {!Host}. *)

val create : ('v, 's) dtype -> int array -> 'v array -> ('v, 's, host) t
(** [create dt s vs] is the host value of shape [s] holding [vs] in C order,
    a float stored as {!Dtype.of_float} says.

    Raises [Invalid_argument] if an extent is negative, [Array.length vs] is not
    the product of [s], or an element is an [int] outside [dt]'s range. *)

val init : ('v, 's) dtype -> int array -> (int array -> 'v) -> ('v, 's, host) t
(** [init dt s f] is the host value of shape [s] whose element at index [i] is
    [f i]. [f] is called once per index, in C order, each time with a fresh
    array.

    Raises [Invalid_argument] as {!create} does. *)

val of_bigarray :
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t -> ('v, 's, host) t
(** [of_bigarray b] is a host value holding a copy of [b]'s elements, of [b]'s
    shape and the dtype that stores its kind. Later writes to [b] do not reach
    it.

    Raises [Invalid_argument] if [b]'s kind is [char], [int] or [nativeint],
    which no dtype stores. *)

(** {1:reads Reads}

    A read gives a value's elements to OCaml. It works at every brand: it waits
    for the work the value depends on, copies its elements to the host and
    leaves every placement as it was. A value of every set is computed on the
    host. A read is not an operation: no interpretation receives it.

    Each raises [Invalid_argument] naming itself for a dead value, and for a
    traced one: [Nx.item: the value is traced by Rune.jit]. *)

val to_array : ('v, 's, 'd) t -> 'v array
(** [to_array x] is [x]'s elements in C order. *)

val item : int list -> ('v, 's, 'd) t -> 'v
(** [item i x] is [x]'s element at [i], one position per axis, a negative one
    counting from the end; [item [] x] reads a 0-d value.

    Raises [Invalid_argument] if [i] does not have one position per axis of [x]
    or a position is outside its axis. *)

val to_bigarray :
  ('v, 's) Bigarray.kind ->
  ('v, 's, 'd) t ->
  ('v, 's, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_bigarray k x] is a fresh bigarray of [x]'s shape holding its elements.
    A dtype Bigarray has no kind for is a type error.

    Raises [Invalid_argument] if [x] has more than 16 axes, Bigarray's most. *)

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

val concatenate : axis:int -> ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [concatenate ~axis xs] is [xs] joined along their axis [axis], in order. A
    copy.

    Raises [Invalid_argument] if [xs] is empty, [axis] is not an axis of its
    first value, or their shapes differ off [axis]. *)

val stack : ?axis:int -> ('v, 's, 'd) t list -> ('v, 's, 'd) t
(** [stack ~axis xs] is [xs] joined along a new axis at position [axis] of the
    result (default [0]), in order. A copy.

    Raises [Invalid_argument] if [xs] is empty, their shapes differ, or [axis]
    is not an axis of the result. *)

val pad : (int * int) array -> 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pad widths v x] is [x] with [fst widths.(i)] elements [v] before its axis
    [i] and [snd widths.(i)] after it. A copy.

    Raises [Invalid_argument] if [widths] does not have one pair per axis of
    [x], a width is negative, or [v] is an [int] outside [x]'s dtype's range. *)

val roll : ?axis:int -> int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [roll ~axis k x] is [x] with its elements along [axis] shifted [k] places
    toward the end, those past it wrapping to the start; a negative [k] shifts
    toward the start. Without [axis], [flatten x] shifted, at [x]'s shape. A
    copy.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

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

(** {1:indexing Indexing and slicing}

    A position written in the program ([I], [L], the ranges) is checked: one
    outside its axis raises, and a negative one counts from the end. A position
    held in data ([T], {!take}, {!take_along_axis}, {!scatter}) never raises and
    never counts from the end: outside [\[0, d)], negative included, it reads
    zero ([+0.], [false], [0 + 0i]) and its write is dropped. That makes a
    gather and an add-scatter adjoint; a program that wants a loud failure
    states it with a check. A [D] start, also held in data, clamps into
    [\[0, d - n\]] instead, so its window lies in the axis. *)

(** The type for selections along one axis. *)
type 'd index =
  | I of int  (** One position; the axis goes. *)
  | L of int list  (** Positions written in the program; a gather. *)
  | T of 'd int64_t
      (** Positions held in data, replacing the axis by their axes; a gather. *)
  | R of int * int
      (** [R (start, stop)]: [start] to [stop - 1]. A negative [start] or [stop]
          counts from the end; then both clip into [\[0, d\]]. *)
  | Rs of int * int * int
      (** [Rs (start, stop, step)]: [start], [start + step], … toward [stop],
          [stop] excluded. A negative [start] or [stop] counts from the end;
          then both clip into [\[0, d\]] for [step > 0] and into
          [\[-1, d - 1\]] for [step < 0], so [Rs (-1, -d - 1, -1)] is the axis
          reversed. *)
  | A  (** The whole axis. *)
  | N  (** A new axis of extent [1]; it selects along no axis of [x]. *)
  | D of 'd int64_t * int
      (** [D (start, n)]: [n] consecutive positions from the 0-d [start] held
          in data, the start clamped into [\[0, d - n\]]; a gather. *)

val slice : 'd index list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [slice idx x] selects from [x] axis by axis: each entry selects along the
    next axis independently of the others, [N] excepted, and axes past the
    list are whole. [I], [R], [Rs], [A] and [N] give a view; [L], [T] and [D]
    gather.

    Raises [Invalid_argument] if [idx] addresses more axes than [x] has, a
    written position is outside its axis, a step is [0], a [D] start is not
    0-d, or a [D] length is negative or exceeds its axis. *)

val get : int list -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [get p x] is [slice (List.map (fun i -> I i) p) x], a view. *)

val take : ?axis:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [take ~axis p x] replaces [x]'s axis [axis] by [p]'s axes, reading [x] at
    each position [p] holds; without [axis], [flatten x] is read. A position
    outside [\[0, d)] reads zero. A gather.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

val take_along_axis :
  axis:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [take_along_axis ~axis p x] has [p]'s shape broadcast with [x]'s off
    [axis]; its element at [j] is [x]'s element at [j] with axis [axis]
    replaced by [p]'s element at [j]. A position outside [\[0, d)] reads zero.
    A gather.

    Raises [Invalid_argument] if [axis] is not an axis of [x], [p]'s rank is
    not [x]'s, or the shapes off [axis] do not broadcast. *)

(** {1:updates Functional updates} *)

val set : 'd index list -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [set idx v x] is [x] with [slice idx x] replaced by [v], broadcast to the
    selection's shape. [x] is unchanged. Without {!donate} it costs a copy of
    [x]; [set idx v (donate x)] writes only the selection, into [x]'s memory,
    where {!donate} lets the consumer write there. A position in [T] outside its
    axis writes nothing; where [T] repeats a position, the last update, in C
    order of the selection, wins. A [D] window clamps its start as {!slice}
    does, so a start past [d - n] writes the last [n] positions: a decode step
    whose position reaches the cache's capacity overwrites its last row, and a
    program that can pass capacity states the bound with a check.

    Raises [Invalid_argument] as {!slice} does, if [v] does not broadcast to
    the selection's shape, or if an [L] names one position twice, counting
    from the end. *)

(** The type for how {!scatter} combines a target with its updates. *)
type combine =
  | Set  (** The last update. *)
  | Add  (** [+0] plus the target plus its updates. *)
  | Max  (** The largest of the target and its updates, the first NaN. *)
  | Min  (** The smallest, the first NaN. *)

val scatter :
  ?combine:combine ->
  axis:int ->
  'd int64_t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [scatter ~combine ~axis p u x] is [x] with each element of [u] combined
    into [x] at its own index with axis [axis] replaced by [p]'s element there;
    [p] and [u] have one shape. A target's updates apply in C order of [u]:
    [Set] (default) keeps the last; [Add] gives, where some update lands, [+0]
    plus the element plus its updates, a narrow float summed in float32 and
    rounded once, and elsewhere the element keeps its bits; [Max] and [Min]
    keep the first NaN. A position outside [\[0, d)] writes nothing. A float
    [Add] gives the same bits on every run.

    Raises [Invalid_argument] if [axis] is not an axis of [x], [p] and [u]
    differ in shape, their shape differs from [x]'s off [axis], or [combine]
    is [Add] and the dtype is boolean. *)

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

val square : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [square x] is [mul x x]. Every dtype but booleans. *)

val recip : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [recip x] is [1 / x]; on integers, [x] for [1] and [-1] and [0] otherwise.
    Every dtype but booleans. *)

val abs : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [abs x] is [x]'s magnitude. Floats and integers. *)

val sign : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sign x] is [-1], [0] or [1] by [x]'s sign, and NaN for a NaN. Floats and
    integers. *)

(** {2:scalars Scalar forms}

    [f_s x c] is [f x (scalar (dtype x) c)] and [rf_s c x] is
    [f (scalar (dtype x) c) x]: the constant takes [x]'s dtype, a float
    stored as {!Dtype.of_float} says. Each raises as its function does, naming
    itself, and [Invalid_argument] if [c] is an [int] outside [x]'s dtype's
    range. *)

val add_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [add_s x c] is [add x c]. *)

val sub_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [sub_s x c] is [sub x c]. *)

val mul_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [mul_s x c] is [mul x c]. *)

val div_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [div_s x c] is [div x c]. *)

val pow_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [pow_s x c] is [pow x c]. *)

val mod_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [mod_s x c] is [mod_ x c]. *)

val maximum_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [maximum_s x c] is [maximum x c]. *)

val minimum_s : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [minimum_s x c] is [minimum x c]. *)

val rsub_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rsub_s c x] is [sub c x]. *)

val rdiv_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rdiv_s c x] is [div c x]. *)

val rpow_s : 'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rpow_s c x] is [pow c x]. *)

val equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [equal_s x c] is [equal x c]. *)

val not_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [not_equal_s x c] is [not_equal x c]. *)

val less_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [less_s x c] is [less x c]. *)

val less_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [less_equal_s x c] is [less_equal x c]. *)

val greater_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [greater_s x c] is [greater x c]. *)

val greater_equal_s : ('v, 's, 'd) t -> 'v -> 'd bool_t
(** [greater_equal_s x c] is [greater_equal x c]. *)

(** {2:operators Operators}

    The arithmetic operators, for a local open: [Nx.(sin x * x +$ 1.)].
    Comparisons and logical operators live in {!Infix}, so that [Nx.( … )]
    keeps OCaml's own. *)

val ( + ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a + b] is [add a b]. *)

val ( - ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a - b] is [sub a b]. *)

val ( * ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a * b] is [mul a b]. *)

val ( / ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a / b] is [div a b]. *)

val ( ** ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [a ** b] is [pow a b]. *)

val ( ~- ) : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [-x] is [neg x]. *)

val ( +$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x +$ c] is [add_s x c]. *)

val ( -$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x -$ c] is [sub_s x c]. *)

val ( *$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x *$ c] is [mul_s x c]. *)

val ( /$ ) : ('v, 's, 'd) t -> 'v -> ('v, 's, 'd) t
(** [x /$ c] is [div_s x c]. *)

(** Comparisons and logical operators, for [open Nx.Infix]. *)
module Infix : sig
  val ( = ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a = b] is [equal a b]. *)

  val ( <> ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a <> b] is [not_equal a b]. *)

  val ( < ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a < b] is [less a b]. *)

  val ( <= ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a <= b] is [less_equal a b]. *)

  val ( > ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a > b] is [greater a b]. *)

  val ( >= ) : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
  (** [a >= b] is [greater_equal a b]. *)

  val ( =$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x =$ c] is [equal_s x c]. *)

  val ( <>$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <>$ c] is [not_equal_s x c]. *)

  val ( <$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <$ c] is [less_s x c]. *)

  val ( <=$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x <=$ c] is [less_equal_s x c]. *)

  val ( >$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x >$ c] is [greater_s x c]. *)

  val ( >=$ ) : ('v, 's, 'd) t -> 'v -> 'd bool_t
  (** [x >=$ c] is [greater_equal_s x c]. *)

  val ( && ) : 'd bool_t -> 'd bool_t -> 'd bool_t
  (** [a && b] is [logical_and a b]. Both operands are computed: it is no
      short circuit. *)

  val ( || ) : 'd bool_t -> 'd bool_t -> 'd bool_t
  (** [a || b] is [logical_or a b], as [( && )]. *)
end

(** {2:transcendental Powers, exponentials and trigonometry}

    Floats only. *)

val sqrt : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sqrt x] is the square root, NaN below [-0.]. *)

val rsqrt : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rsqrt x] is [recip (sqrt x)]: [inf] at [+0.], [-inf] at [-0.], NaN below.
*)

val hypot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [hypot x y] is [sqrt (x{^2} + y{^2})] with no intermediate overflow: finite
    wherever the result is. It is [inf] where [x] or [y] is an infinity, a NaN
    beside it included, and NaN where either is a NaN otherwise. Within
    3 ulps at float32 and float64. *)

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

(* CR: State atan2's mathematical branch as [-pi, pi], then describe its
   floating-point approximation. atan2(-0, -1) gives negative rounded pi;
   float8_e4m3fn encodes the endpoints as +/-3.25, so the stated interval
   cannot bound the outputs. Keep y's signed-zero choice and correct
   Prog.Atan2's point to (x, y), with y still the first operand. *)
val atan2 : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan2 y x] is the angle of the point [(x, y)], in \][-π], [π]\]. *)

val sinh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sinh x] is the hyperbolic sine. *)

val cosh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cosh x] is the hyperbolic cosine. *)

val tanh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tanh x] is the hyperbolic tangent. *)

val asinh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [asinh x] is the inverse hyperbolic sine, odd, [x] itself near [0.]: within
    3 ulps at float32 and float64. *)

val acosh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [acosh x] is the inverse hyperbolic cosine: [0.] at [1.], NaN below [1.],
    within 3 ulps at float32 and float64. *)

val atanh : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atanh x] is the inverse hyperbolic tangent: [inf] at [1.], [-inf] at
    [-1.], NaN beyond them, odd, [x] itself near [0.]; within 3 ulps at
    float32 and float64. *)

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

val bitwise_not : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [bitwise_not x] is every bit of [x] flipped, the logical not of booleans.
    Integers and booleans. *)

val lshift : ('v, 's, 'd) t -> int -> ('v, 's, 'd) t
(** [lshift x n] is [x] shifted [n] bits toward its high end: [x 2{^n}] modulo
    [2{^w}] for a dtype of [w] bits, [0] once [n] reaches [w]. Integers.

    Raises [Invalid_argument] if [n < 0]. *)

val rshift : ('v, 's, 'd) t -> int -> ('v, 's, 'd) t
(** [rshift x n] is [x] shifted [n] bits toward its low end: [x / 2{^n}]
    rounded toward negative infinity, so a signed [x] keeps its sign; [0], or
    [-1] for a negative [x], once [n] reaches the width. Integers.

    Raises [Invalid_argument] if [n < 0]. *)

val logical_and : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_and a b] is one where [a] and [b] are both not zero, zero
    elsewhere, in their dtype. A NaN is not zero. Every dtype. *)

val logical_or : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_or a b] is one where [a] or [b] is not zero, as {!logical_and}. *)

val logical_xor : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_xor a b] is one where exactly one of [a] and [b] is not zero, as
    {!logical_and}. *)

val logical_not : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [logical_not x] is one where [x] is zero, zero elsewhere, as
    {!logical_and}. *)

val isnan : ('v, 's, 'd) t -> 'd bool_t
(** [isnan x] is [true] where [x] is a NaN, or a complex number with a NaN
    part. Every dtype: [false] for the integers and booleans. *)

val isinf : ('v, 's, 'd) t -> 'd bool_t
(** [isinf x] is [true] where [x] is an infinity, or a complex number with an
    infinite part. Every dtype: [false] for the integers, the booleans and the
    float formats without infinities. *)

val isfinite : ('v, 's, 'd) t -> 'd bool_t
(** [isfinite x] is [true] where [x] is neither an infinity nor a NaN, and for
    a complex number where both parts are. Every dtype: [true] for the integers
    and booleans. *)

val clamp : ?min:'v -> ?max:'v -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [clamp ~min ~max x] is [minimum (maximum x min) max] with each bound a
    constant, a bound left out not applied: [max] where [min > max], and a NaN
    stays a NaN. [x] itself without either. Every dtype.

    Raises [Invalid_argument] if a bound is an [int] outside [x]'s dtype's
    range. *)

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

(** {1:contraction Contraction}

    A contraction multiplies two operands along their paired axes and sums the
    products over the summed ones. *)

val contract :
  ?sizes:(string * int) list ->
  ?acc:('w, 'q) dtype ->
  ?init:('v, 's, 'd) t ->
  ('v, 's) dtype ->
  Pattern.t ->
  ('a, 'b, 'd) t ->
  ('c, 'e, 'd) t ->
  ('v, 's, 'd) t
(** [contract dt p a b] is [round_dt (init + Σ a · b)] over [p]'s summed names,
    summed in [acc], for each index of [p]'s result. [sizes] gives the extents
    of names inside groups that neither operand's shape gives alone.

    [acc] defaults to [float32] for float operands of 32 bits or fewer,
    [float64], [complex64] or [complex128] at the operands' width otherwise, and
    [dt] for integers, which wrap. An integer operand with a float one has no
    default: [acc] is required. The sum starts from [init], or [+0] without it,
    and associates in an order that is a function of the shapes alone, never of
    the order a pattern lists its names, which each kernel library states.
    Before rounding it is within [γ(K + 1, 2u) (|init| + Σ|a||b|)] of the exact
    sum of its [K] products, where [γ(n, v) = n v / (1 - n v)] and [u] is
    [acc]'s unit roundoff.

    A donated [init] is consumed as {!donate} states.

    Raises [Invalid_argument] if [dt] is a boolean; [p] has one operand; an
    operand's rank is not the number of axes [p] gives it; paired extents
    differ; a group's extents do not multiply to its axis, or it leaves two
    extents unknown; a unit axis's extent is not [1]; a name in [sizes] is not
    in [p]; [acc] is omitted for an integer and a float operand, or is a
    boolean, a float narrower than [float32] or than a float operand, or of
    another kind than [dt]; or [init]'s shape is not the result's. *)

val einsum : Pattern.t -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [einsum p a b] is [contract (dtype a) p a b]: two operands, as a contraction
    has. A product of three is two calls.

    Raises [Invalid_argument] as {!contract} does. *)

val matmul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [matmul a b] is the matrix product over the last two axes, leading axes
    broadcast, at [a]'s dtype and accumulated as {!contract} does by default. A
    1-d [a] is a row and a 1-d [b] a column, and their unit axis is dropped from
    the result: two 1-d operands give their 0-d inner product.

    Raises [Invalid_argument] if an operand is 0-d, the inner extents differ,
    the leading axes do not broadcast, or the dtype is a boolean. *)

(** {1:devices Device sets and placement}

    A value lies on a device set, a module minted by {!devices} whose brand ['d]
    keeps its values apart from other sets'. Within its set a value has a
    placement: whole on one or every device, or cut into windows across them.
    {!place} moves a value between sets, and between placements of one. The host
    is a set like any other, {!Host}. *)

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
  type +'d t
  (** The type for placements over a set of brand ['d]. *)

  val on : 'd devices -> 'd t
  (** [on s] holds the whole value on every device of [s]. *)

  val device : 'd devices -> Rig.t -> 'd t
  (** [device s d] holds the whole value on [d] alone, a device of [s];
      [on s] where [s] has [d] alone.

      Raises [Invalid_argument] naming [Nx.Placement.device] unless [d]
      is a device of [s] ({!Rig.equal}). *)

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

  val of_shards :
    'd Placement.t -> ('v, 's) Nx_array.t iarray -> ('v, 's, 'd) t
  (** [of_shards p arrays] is the value whose bytes are [arrays], one per
      device of [p] in order, each that device's window.

      Raises [Invalid_argument] naming [Nx.Repr.of_shards] unless there
      is one array per device of [p], each on its device, all of one
      shape, of a rank that has every axis [p] cuts. *)

  val shards : ('v, 's, 'd) t -> ('v, 's) Nx_array.t iarray option
  (** [shards x] is [Some arrays], one per device of [x]'s placement, in
      order: [[| a |]] for a value on one device; [None] for a value of
      every set. The arrays are for reading, as {!array}'s. *)
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
    | Gather : {
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) nx;
        x : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** [x] read at the positions [idx] holds along [axis]: [idx] has [x]'s
            rank and its extents off [axis], and the result [idx]'s shape. A
            position outside [x]'s axis reads the element of zero bits. *)
    | Scatter : {
        combine : Nx_kernel.Spec.combine;
        unique : bool;
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) nx;
        updates : ('v, 's, 'd) nx;
        into : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** [into] with each update combined at its own index with axis [axis]
            replaced by [idx]'s element there, in C order of [updates]
            ({!Nx_kernel.Spec.scatter}); [idx] and [updates] have one shape,
            [into]'s off [axis]. [unique] promises distinct targets. *)
    | Assemble : {
        dtype : ('v, 's) dtype;
        shape : int array;
        fill : 'v;
        pieces : (Nx_array.Move.range array * ('v, 's, 'd) nx) list;
      }
        -> ('v, 's, 'd) nx t
        (** The value of [shape] whose element at an index is the last piece's
            whose region, a [Slice] of [shape] by its ranges, holds it, and
            [fill] where none does. Each piece has its region's shape. *)
    | Contract : {
        spec : Nx_kernel.Spec.contract Nx_kernel.Spec.t;
        out : ('v, 's) dtype;  (** [spec]'s [out]. *)
        a : ('a, 'b, 'd) nx;
        b : ('c, 'e, 'd) nx;
        init : ('v, 's, 'd) nx option;  (** Present iff [spec] has one. *)
      }
        -> ('v, 's, 'd) nx t
        (** [spec] of [a] and [b], from [init]: its result C-contiguous, of the
            shape {!Nx_kernel.Spec.shapes} gives. *)
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
      operands [x0], [x1], …, and each operand's dtype, shape and placement. A
      program node read more than once prints once, as [nK = …] before the
      outputs, and as [nK] where it is read. *)

  val operands : 'r t -> operands
  (** [operands op] is [op]'s operands in order: a map's loads, a contraction's
      [a], [b] then [init], [Check]'s [ok] then its data, the one operand of the
      others. *)

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
