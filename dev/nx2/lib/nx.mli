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
    every handle of [x] die: a later operation on any of them raises
    [Invalid_argument] naming the consumer, and its shape, dtype and placement
    still answer.

    A movement, which rearranges elements without computing them ({!reshape},
    {!transpose}, {!flip}, {!broadcast_to}, slicing by ranges,
    {!sliding_window}), spends its handle and passes a new one to its result,
    whether it makes a view or a copy; the deaths wait for the final consumer. A spent
    handle raises naming the movement that spent it. Any other operation
    consumes its handles. Two handles of one donor or over one memory reaching
    one operation raise naming [Nx.donate]: the same handle twice, two donations
    of one value, or a handle and one a movement passed on.

    An operation checks every operand before it consumes any, so in sequence a
    refused operation kills nothing. On any number of domains, no operation
    reads a value after its donor died, and no value is consumed twice. When
    operations on several domains race to consume the same handles, at least
    one succeeds. A loser raises naming an operation that claimed one of its
    operands, which may itself have lost, and the loser's other donated operands
    may be dead. No operation reads elements a consumer has overwritten.

    The consumer may write its result into the donated memory when it reads the
    operand only at the result's own index (an elementwise operand), the memory
    is C-contiguous with the result's shape and dtype, and the memory is not
    shared. Memory is shared once a value outside the donor's handles has been
    made over it (a view, a borrow by {!place}, a {!Repr} crossing), and stays
    shared. Otherwise the consumer computes into fresh memory, and a value that
    shares the memory keeps it. Its result is the same either way.

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

val map_item : ('v -> 'v) -> ('v, 's, 'd) t -> ('v, 's, host) t
(** [map_item f x] is the host value of [x]'s shape whose element at each index
    is [f] of [x]'s element there, a float stored as {!Dtype.of_float} says.
    [f] is an OCaml function, so it runs on the host, once per element in C
    order. [x], at any brand, is read as {!to_array} reads it.

    Raises [Invalid_argument] if [f] gives an [int] outside [x]'s dtype's
    range, and as {!to_array} does. *)

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
    for an extent [d], or [0] where [window] exceeds [d], and an axis of extent
    [window] is appended: window [w]'s element [j] is [x]'s element
    [w * step + j] along [axis]. A view, whose windows share elements where
    [step < window].

    Raises [Invalid_argument] if [axis] is not [x]'s, or [window] or [step] is
    below [1]. *)

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
    with one operand, {!contract}, with two, and {!einsum}, with two or more:

    {v
    pattern  ::= operands "->" layout [ "|" name { name } ]
    operands ::= layout { "," layout }
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

      With two or more, a name in the result is a batch axis where two operands
      or more hold it and a free axis where one does, and the names after [|]
      are summed: each is in two operands or more and not in the result. Every
      operand name is in the result or summed. A group in an operand splits
      that operand's axis, and an extent a group leaves unknown comes from
      another operand that holds the name. [...] stands for the same leading
      axes in every layout, kept as batch axes; [1] drops a unit axis of an
      operand or adds one to the result.

      Sums are written out because a name dropped from the result is easy to
      miss. In ["b h q d, b h k d -> h q k | d"], [b] was left out by mistake;
      summing it would give a result of plausible shape and wrong values, so
      the pattern raises.

      Raises [Invalid_argument] naming [s] if it is not in the grammar, a name
      repeats within a layout, [...] appears twice in a layout, or a name breaks
      the rules above. A string in NumPy's einsum notation, such as
      ["bik,bkj->bij"], raises with the pattern it means:
      ["b i k, b k j -> b i j | k"]. *)

  val inverse : t -> t
  (** [inverse p] is the one-operand pattern that undoes [p]: its sides
      exchanged.

      Raises [Invalid_argument] if [p] has more than one operand. *)
end

val rearrange :
  ?sizes:(string * int) list -> Pattern.t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [rearrange ~sizes p x] is [x] split, permuted and merged as [p] says: the
    result's element at an index is [x]'s element where each name takes the same
    position. A name alone on the left takes its axis's extent, and [sizes]
    gives others; a group on the left may hold one name whose extent neither
    gives, the quotient of its axis by the others. A view where strides express
    it, a copy otherwise.

    Raises [Invalid_argument] if [p] has more than one operand, [p]'s left side
    names more axes than [x] has or, without [...], other than [x]'s rank, a
    [1] on the left meets an extent other than [1], a group's extents do not
    divide or multiply to its axis, a group has two unknown extents, a size
    disagrees with the extent [x] gives its name, a name in [sizes] is not in
    [p] or is there twice, or a size is negative. *)

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
    stated with it. The host computes every dtype stated. Where a device's
    kernels do not compute one, as a GPU's may not compute complex numbers,
    the function raises naming itself. *)

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

    Floats only. Each is its mathematical value rounded to the dtype;
    a stated range is the mathematical one, which rounding can leave
    at an end: in [float8_e4m3fn], [π] rounds to [3.25]. *)

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

val atan2 : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [atan2 y x] is the angle of the point [(x, y)], in \[[-π], [π]\]:
    [-π] for [y = -0.] and [x < 0]. *)

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

(** {2:special Special functions}

    The laws of the distributions {!Rng} samples. Arguments broadcast,
    parameters first and the variable last: [gammainc a] is the distribution
    function of the gamma distribution of shape [a]. Outside a function's
    domain, and at a NaN argument, the value is NaN; at a pole it is the
    function's limit there.

    Each function states a bound at float32 and float64 on its {e error}, the
    number of floats of the dtype between its result and the exact value
    rounded to the dtype. A bound of [k] ulps {e absolute near zeros} also
    holds where the exact value is below [1.] in magnitude and the result
    within [kε] of it, [ε] being the distance from [1.] to the next float of
    the dtype. An inverse's bound adds [κ] times its forward function's bound,
    where [κ = |p / (x F'(x))|] at its result [x], for [F] the forward
    function, is its condition number. A subnormal result is held to its
    bound in units of the least subnormal, plus half of one.

    The formats narrower than 32 bits compute at [float32] and round once.
    Where a bound is in ulps, their result is within one ulp of the exact
    value rounded to their dtype; where it is absolute, within that bound at
    [float32] plus half an ulp of their dtype.

    A device that flushes subnormal numbers to zero reads a subnormal operand
    as zero and writes a subnormal result as zero; there each bound holds
    where the operands and the result are normal. *)

val erfc : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erfc x] is [1 - erf x], to its own relative precision where [erf x]
    rounds to [1.]: [erfc 10.] is [2.088e-45] at float64. It is [2.] at
    [-inf] and [0.] at [inf]. Within 8 ulps. *)

val erfinv : (float, 's, 'd) t -> (float, 's, 'd) t
(** [erfinv p] is the [x] with [erf x = p], on \[[-1.], [1.]\]: [-inf] at
    [-1.] and [inf] at [1.]. Within [4 + 8κ] ulps, where
    [κ = |p √π e{^x²} / (2x)|] is [1] at [0.] and grows as [p] nears [±1.]. *)

val ndtr : (float, 's, 'd) t -> (float, 's, 'd) t
(** [ndtr x] is the standard normal distribution function
    [Φ(x) = erfc (-x/√2) / 2]. Its lower tail keeps its relative precision
    down to the least subnormal, at [x] about [-38.5] at float64 and [-14.]
    at float32. It is [0.] at [-inf] and [1.] at [inf]. Within 16 ulps. *)

val log_ndtr : (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_ndtr x] is [log (ndtr x)], finite where [ndtr x] underflows while
    [x²/2] is finite: down to [x] about [-1.9e154] at float64 and [-2.6e19]
    at float32, and [-inf] below. It is [-inf] at [-inf] and [-0.] at [inf].
    Within 32 ulps. *)

val ndtri : (float, 's, 'd) t -> (float, 's, 'd) t
(** [ndtri p] is the [x] with [ndtr x = p], on \[[0.], [1.]\]: [-inf] at [0.]
    and [inf] at [1.]. A [p] above [1/2] is read as one minus the upper tail,
    so the result keeps the precision of [p] near either end:
    [ndtri 1e-300] is [-37.047]. Within [4 + 16κ] ulps, where
    [κ = |p / (x Φ'(x))|]. *)

val lgamma : (float, 's, 'd) t -> (float, 's, 'd) t
(** [lgamma x] is [log |Γ(x)|], on every float: [inf] at [0.], the negative
    integers and [inf], NaN at [-inf]. For [x > 0] within 16 ulps, absolute
    near its zeros at [1.] and [2.]; below [0.], within 16 ulps or
    [16ε (1 + lgamma (1 - x))] absolutely, the precision its reflection
    [log π - log |sin πx| - lgamma (1 - x)] keeps. *)

val digamma : (float, 's, 'd) t -> (float, 's, 'd) t
(** [digamma x] is [ψ(x) = Γ'(x) / Γ(x)], the mean of [log X] for [X] a
    gamma draw of shape [x], on every float: [-inf] at [+0.], [inf] at [-0.]
    and [inf], NaN at the negative integers and [-inf]. For [x > 0] within
    16 ulps, absolute near its root at [1.4616]; below [0.], within 16 ulps
    or [16ε (1 + |π cot πx|)] absolutely. *)

val lbeta : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [lbeta a b] is [log B(a, b) = lgamma a + lgamma b - lgamma (a + b)] for
    [a, b ≥ 0], computed without that sum's cancellation: [lbeta 1e10 1.] is
    [-log 1e10]. It is [inf] where either is [0.] and the other finite,
    [-inf] where either is [inf] and the other positive, and NaN where one is
    [0.] and the other [inf]. Within 256 ulps, or [512ε] absolutely where it
    is below [1.] in magnitude. *)

val gammainc : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammainc a x] is the regularised lower incomplete gamma function
    [P(a, x) = (1/Γ(a)) ∫₀ˣ t{^a-1} e{^-t} dt], the distribution function at
    [x] of the gamma distribution of shape [a] and unit rate, for [a, x ≥ 0].
    It is [0.] at [x = 0.] and where [a] is [inf] for finite [x], [1.] at
    [x = inf] and, its limit, at [a = 0.] for [x > 0]; NaN at [a = x = 0.]
    and [a = x = inf], where the limits disagree. For [0 < a ≤ 2{^20}]
    within [32 (1 + |log P|) + 4] ulps; above, its error grows with [a]. *)

val gammaincc : (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammaincc a x] is [Q(a, x) = 1 - P(a, x)] ({!gammainc}), the upper
    tail, to its own relative precision where [P] rounds to [1.]:
    [gammaincc 1. 40.] is [e{^-40}]. Its edges are the complements of
    {!gammainc}'s. Within [32 (1 + |log Q|) + 4] ulps for
    [0 < a ≤ 2{^20}]. *)

val log_gammainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_gammainc a x] is [log P(a, x)] ({!gammainc}), finite for [a, x > 0]
    where [P] underflows: [log_gammainc 1000. 1.] is [-5913.13]. It is [-inf]
    at [x = 0.] and where [a] is [inf], [0.] at [x = inf], and [-0.] at
    [a = 0.] for [x > 0], where [P] reaches [1.] from below. For
    [0 < a ≤ 2{^20}] within 16 ulps, absolute near zeros. *)

val log_gammaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_gammaincc a x] is [log Q(a, x)] ({!gammaincc}), finite for [a > 0]
    wherever [x] is. It is [0.] at [x = 0.], [-0.] where [a] is [inf] for
    [x > 0], where [Q] reaches [1.] from below, and [-inf] at [x = inf] and
    at [a = 0.]. Within the bound of {!log_gammainc}. *)

val gammaincinv :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammaincinv a p] is the [x] with [gammainc a x = p], the gamma
    distribution's quantile, for [a ≥ 0] and [p] in \[[0.], [1.]\]. A [p]
    above [1/2] is read as one minus the upper tail, so the result keeps its
    precision near either end. It is [0.] at [p = 0.], [inf] at [p = 1.] and
    where [a] is [inf] for [p > 0], and at [a = 0.], where the mass is at
    [0.], [0.] for [p < 1]. For [0 < a ≤ 2{^20}] within
    [4 + κ (32 (1 + |log p|) + 4)] ulps, where [κ = |p / (x ∂ₓP(a, x))|]. *)

val gammainccinv :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [gammainccinv a q] is the [x] with [gammaincc a x = q], as
    {!gammaincinv} from the upper tail: [inf] at [q = 0.], [0.] at [q = 1.],
    and [0.] at [a = 0.] for [q > 0]. Within the bound of {!gammaincinv},
    in [q]. *)

val betainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [betainc a b x] is the regularised incomplete beta function
    [I_x(a, b) = (1/B(a, b)) ∫₀ˣ t{^a-1} (1 - t){^b-1} dt], the distribution
    function at [x] of the beta distribution of shapes [a] and [b], for
    [a, b ≥ 0] and [x] in \[[0.], [1.]\]. It is [0.] at [x = 0.] and [1.] at
    [x = 1.]; its limits, [1.] at [a = 0.] and where [b] is [inf], and [0.]
    at [b = 0.] and where [a] is [inf]; NaN where two of these disagree. For
    [0 < a, b ≤ 2{^20}] within [64 (1 + |log I|) + 4] ulps; above, its
    error grows with [a] and [b]. *)

val betaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [betaincc a b x] is [1 - I_x(a, b)] ({!betainc}), the upper tail, to its
    own relative precision where [I] rounds to [1.]: [betaincc 1. 60. 0.5] is
    [2{^-60}]. For [x ≥ 1/2] it is [betainc b a (1 - x)] bit for bit. Its
    edges are the complements of {!betainc}'s. Within
    [64 (1 + |log (1 - I)|) + 4] ulps for [0 < a, b ≤ 2{^20}]. *)

val log_betainc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [log_betainc a b x] is [log I_x(a, b)] ({!betainc}), finite for finite
    [a, b > 0] and [x] in (0, 1\], where {!betainc} underflows included. It
    is [-inf] where {!betainc} is [0.], [0.] at [x = 1.], and [-0.] at
    [a = 0.] and where [b] is [inf] for [x] in (0, 1), where the limit is
    reached from below. For [0 < a, b ≤ 2{^20}] within 32 ulps, absolute
    near zeros. *)

val log_betaincc :
  (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [log_betaincc a b x] is [log (1 - I_x(a, b))] ({!betaincc}), finite for
    finite [a, b > 0] and [x] in \[0, 1). It is [-inf] where {!betaincc} is
    [0.], [0.] at [x = 0.], and [-0.] at [b = 0.] and where [a] is [inf] for
    [x] in (0, 1). Within the bound of {!log_betainc}. *)

val i0e : (float, 's, 'd) t -> (float, 's, 'd) t
(** [i0e x] is [e{^-|x|} I₀(x)], [I₀] the modified Bessel function of the
    first kind of order 0: even, [1.] at [0.] and [0.] at [±inf]. The von
    Mises normaliser [2π I₀(k)] has the logarithm [k + log (2π i0e k)], finite
    where [I₀(k)] overflows, past [k] about [714] at float64 and [92] at
    float32. Within 8 ulps. *)

val i1e : (float, 's, 'd) t -> (float, 's, 'd) t
(** [i1e x] is [e{^-|x|} I₁(x)], [I₁] the modified Bessel function of the
    first kind of order 1: odd, so [-0.] at [-0.] and at [-inf], and [0.] at
    [inf]. [i1e k / i0e k] is the mean of [cos θ] under the von Mises
    distribution of concentration [k]. Within 8 ulps. *)

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

(** {2:complex Complex numbers}

    A complex value's parts are floats of half its width: float32 for
    complex64, float64 for complex128. {!real}, {!imag}, {!magnitude} and
    {!angle} take the dtype of their result first. *)

val real : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [real dt z] is [z]'s real parts, stored in [dt] by the conversion rule
    ({!cast}): exact where [dt] is as wide as the parts. *)

val imag : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [imag dt z] is [z]'s imaginary parts, as {!real}. *)

val magnitude : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [magnitude dt z] is [z]'s modulus, {!hypot} of its parts computed in the
    wider of [dt] and the parts' format, then stored in [dt]. *)

val angle : (float, 'r) dtype -> (Complex.t, 's, 'd) t -> (float, 'r, 'd) t
(** [angle dt z] is [z]'s argument in radians, {!atan2} of its imaginary and
    real parts, computed as {!magnitude} is. The negative real axis is a branch
    cut whose side the sign of a zero imaginary part picks: [-1 + 0i] has
    angle [π], [-1 - 0i] has [-π]; [0] has angle [0]. *)

val complex :
  (Complex.t, 'c) dtype ->
  re:(float, 'a, 'd) t ->
  im:(float, 'a, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [complex dt ~re ~im] is the complex value of parts [re] and [im],
    broadcast together, each stored in [dt]'s part format by the conversion
    rule: infinities, NaN and signed zeros stay, and one part never affects the
    other. [complex dt ~re:(real f z) ~im:(imag f z)] is [z] for [f] the
    parts' format.

    Raises [Invalid_argument] if the shapes do not broadcast. *)

val conjugate : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [conjugate x] is [x] with each imaginary part negated, [0.] to [-0.]
    included; [x] itself for a dtype that is not complex. *)

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

val array_equal : ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd bool_t
(** [array_equal a b] is the 0-d [true] iff [a] and [b] have one shape and
    {!equal} holds at every index: [false] where an element is a NaN, and
    [true] for two values without elements. Values of two shapes give [false]
    and do not broadcast. *)

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
    float result that a NaN reaches is the first NaN in C order. A complex
    number with a NaN part is a NaN to {!prod}, {!max}, {!min}, their
    positions and scans; {!sum} and {!cumsum} add the parts apart. The float
    formats narrower than 32 bits accumulate at [float32] and round once.

    Each raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated, or a dtype it does not take. The host computes every dtype
    stated. Where a device's kernels do not compute one, as a GPU's may not
    compute complex numbers, the function raises naming itself. *)

val sum : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [sum ~axes x] adds the elements from [+0]: a sum of [-0.] terms is [+0.],
    and a sum of none is [+0.]. Integers wrap. Every dtype but booleans. *)

val prod : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [prod ~axes x] multiplies the elements from the first, as {!sum}, and a
    product of none is [1]. Floats, integers and complex numbers. *)

val max : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [max ~axes x] is the greatest element, as {!maximum} orders them: [-0.]
    below [+0.], any on booleans. Every dtype.

    Raises [Invalid_argument] where an axis it reduces is empty. *)

val min : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [min ~axes x] is the least element, as {!max}; all on booleans. *)

val mean : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [mean ~axes x] is the {!sum} divided by the number of elements, rounded
    once: NaN for none. Floats and complex numbers. *)

val all : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd bool_t
(** [all ~axes x] is [true] where every element is non-zero, as {!cast} into
    {!bool} reads it: a NaN is non-zero, and a complex number is where a part
    is. Of no element it is [true]. Every dtype. *)

val any : ?axes:int list -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd bool_t
(** [any ~axes x] is [true] where some element is non-zero, as {!all}. Of no
    element it is [false]. *)

val count : ?axes:int list -> ?keepdims:bool -> (bool, 's, 'd) t -> 'd int64_t
(** [count ~axes m] is the number of [true] elements of [m], a {!bool} or a
    {!bit} value: [0] of none. *)

val argmax : ?axis:int -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd int64_t
(** [argmax ~axis x] is the position along [axis] of the first element
    greatest as {!max} orders them: the first NaN where there is one, and
    [1] for [[-0.; 0.]]. Without [axis], the position in C order among all of
    [x]'s elements, of rank [0], or every axis of extent [1] where
    [keepdims]. Every dtype.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank,
    or where there is no element to choose from. *)

val argmin : ?axis:int -> ?keepdims:bool -> ('v, 's, 'd) t -> 'd int64_t
(** [argmin ~axis x] is the position of the first element least as {!min}
    orders them: the first NaN where there is one, and [1] for [[0.; -0.]].
    As {!argmax} otherwise. *)

val var :
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [var ~axes ~ddof x] is the variance [Σ (x - m)² / (n - ddof)] of the [n]
    elements, [m] their mean; [ddof] defaults to [0], the population
    variance, and [1] gives the sample variance. NaN of no element.

    The population variance is the kernels' moments, within the bound
    their library states; nx's own, the mean then the deviations [d] from
    it and their squares, each summed as {!sum} sums, and
    [(Σ d² - Σ d (Σ d / n)) / n], is within
    [(γ(n + 4) + 2γ(n)²) V + 2γ(n + 6) γ(n) M √V + 4γ(n + 4) γ(n)² M²] of
    the exact [V], [M = Σ|x| / n], barring overflow and underflow,
    [γ(k) = k u / (1 - k u)] and [u] the unit roundoff of the format it
    computes in, and [+∞] where [Σ d²] overflows. A [ddof] scales it by a
    relative [2u] more, then one rounding to the dtype.

    Raises [Invalid_argument] naming itself for a negative [ddof], or one of
    at least [n] where [n > 0]. *)

val std :
  ?axes:int list ->
  ?keepdims:bool ->
  ?ddof:int ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [std ~axes ~ddof x] is the standard deviation, the square root of
    {!var}, taken before the one rounding to the dtype. As {!var} otherwise. *)

val logsumexp :
  ?axes:int list -> ?keepdims:bool -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [logsumexp ~axes x] is [log Σ exp x], computed as
    [m + log Σ exp (x - m)] for the elements' greatest [m], so no term
    overflows: [-∞] of no element or of elements all [-∞], [+∞] where an
    element is [+∞] and none is NaN. Its error is the bound the kernels'
    library states; nx's own, for [n] elements whose greatest is finite, is
    within [γ(2n) + u|r| + ε] of the exact [r], barring overflow and
    underflow, [ε] the error {!exp} and {!log} add. *)

val logmeanexp :
  ?axes:int list -> ?keepdims:bool -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [logmeanexp ~axes x] is [log (Σ exp x / n)]: {!logsumexp} less [log n],
    taken before the one rounding to the dtype. NaN of no element. *)

val cumsum : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumsum ~axis x] is, at each index, the sum of [x]'s elements along [axis]
    up to that index, inclusive; without [axis], of its elements in C order up
    to that one, in [x]'s shape. Every dtype but booleans.

    The result at an index is a function of the elements up to it alone: the
    cumsum of a prefix of the axis is the prefix of the cumsum, bit for bit.
    Its last element need not equal {!sum}'s of the same elements.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank or a
    dtype it does not take. *)

val cumprod : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cumprod ~axis x] is the running product, as {!cumsum}. Floats, integers
    and complex numbers, as {!prod}. *)

val cummax : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummax ~axis x] is the running {!max}, as {!cumsum}. Every dtype. *)

val cummin : ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cummin ~axis x] is the running {!min}, as {!cummax}. *)

val ewma : ?axis:int -> alpha:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [ewma ~alpha x] is the exponentially weighted moving average of [x]
    along [axis]: [y₀ = x₀], then [yᵢ = (1 - alpha) yᵢ₋₁ + alpha xᵢ];
    without [axis], of its elements in C order, in [x]'s shape.

    The average of a prefix of [x] is the prefix of its average, bit for
    bit. Element [i] passes through at most [max(0, 2⌈log₂(i+1)⌉ - 1)]
    compositions of the recurrence's affine maps [y ↦ a y + b], in [O(n)]
    work for [n] elements. A NaN reaches every later element,
    and an infinity as the recurrence carries it; a finite term whose weight
    underflows to [0] drops out.

    Raises [Invalid_argument] naming itself unless [0 < alpha <= 1], or for
    an axis outside [x]'s rank. *)

(** {2:activations Activations}

    Floats only, each computed in [float32] for the formats narrower than 32
    bits and rounded once. *)

val sigmoid : (float, 's, 'd) t -> (float, 's, 'd) t
(** [sigmoid x] is [1 / (1 + exp (-x))], in [[0, 1]]: [+0.] and [1.] only
    where the exact value rounds there, a subnormal where it is one. Within
    3 ulps at float32 and float64. *)

val softplus : (float, 's, 'd) t -> (float, 's, 'd) t
(** [softplus x] is [log (1 + exp x)], computed as
    [max x 0 + log1p (exp (-|x|))]: [exp x] to the format's precision far
    below [0.], [x] far above it, never overflowing. Within 3
    ulps at float32 and float64. *)

val softmax :
  ?axes:int list -> ?scale:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [softmax ~axes ~scale x] is [exp (scale x) / Σ exp (scale x)] along
    [axes], [[-1]] by default; [scale] defaults to [1.]. The exponents are
    shifted by the greatest [scale x] of each lane, so none overflows. A
    lane holding a NaN, or whose greatest [scale x] is infinite, is NaN.

    Raises [Invalid_argument] naming itself for an axis outside [x]'s rank
    or repeated. *)

val log_softmax :
  ?axes:int list -> ?scale:float -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [log_softmax ~axes ~scale x] is the logarithm of {!softmax}, computed as
    [(scale x - m) - log Σ exp (scale x - m)] for the lane's greatest [m]:
    an entry far below [m] keeps its digits where [log (softmax x)] would be
    [-∞]. As {!softmax} otherwise. *)

(** {2:conversion Conversion} *)

val cast : ('w, 'r) dtype -> ('v, 's, 'd) t -> ('w, 'r, 'd) t
(** [cast dt x] is [x]'s elements stored in [dt]; [x] itself where [dt] is [x]'s
    dtype.

    A float rounds once to the nearest value of a float format, ties to even.
    Past the format's largest finite value it is an infinity in [float64],
    [float32], [float16], [bfloat16] and [float8_e5m2], and saturates in the
    formats with none, [float8_e4m3fn] and [float4_e2m1fn]. NaN stays NaN, but in [float4_e2m1fn], where it is [+0.]. Into an
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

    Raises [Invalid_argument] if [dt] is a boolean; [p] has other than two
    operands; an operand's rank is not the number of axes [p] gives it; [...]
    covers different numbers of axes in the two operands; paired extents
    differ; a group's extents do not multiply to its axis, or it leaves
    two extents unknown; a unit axis's extent is not [1]; a name in [sizes] is
    not in [p], or its size is negative; [acc] is omitted for an integer and a
    float operand, or is a boolean, a float narrower than [float32] or than a
    float operand, or of another kind than [dt]; or [init]'s shape is not the
    result's. *)

val einsum :
  ?sizes:(string * int) list ->
  ?acc:('w, 'q) dtype ->
  Pattern.t ->
  ('v, 's, 'd) t list ->
  ('v, 's, 'd) t
(** [einsum ~sizes ~acc p [x1; …; xn]] is [round (Σ x1 ⋯ xn)] over [p]'s
    summed names, for each index of [p]'s result, at the operands' dtype. It
    is a tree of {!contract}s of two operands, each summing in [acc] and
    leaving its value in [acc], so that the result rounds once. [acc] defaults
    as {!contract}'s for two operands of that dtype, and [sizes] is as
    {!contract}'s.

    Of the trees whose values have at most {!Nx_array.Layout.max_rank} axes,
    the tree is the one whose contractions do the fewest multiply-adds in
    all, a contraction doing the product of the extents of the names its two
    parts hold. A part's value holds the names that the result or an operand
    outside it needs, and each of its elements costs at least one
    multiply-add, so the work bounds the memory too. Of trees doing as many,
    each contraction's left part holds the earliest of its operands; of such
    parts, the larger, then the one holding the earliest operand the other
    lacks: equal work gives [((x1 x2) x3) x4]. The tree, and with it the
    result's bits, is a function of [p] and the shapes alone. Choosing it
    takes time in [3^n], so [p] has at most 11 operands.

    {[
    (* For square matrices, a (b v): j·k multiply-adds, then i·j, where
       (a b) v does i·j·k, then i·k. *)
    let mmv = Nx.Pattern.v "i j, j k, k -> i | j k"
    let y = Nx.einsum mmv [ a; b; v ]
    ]}

    Before rounding the result is within [γ(N, 2u) Σ |x1| ⋯ |xn|] of the exact
    sum, where [N = Σ (K + 1)] over the tree's [n - 1] contractions, each
    summing [K] products per element, and [γ] and [u] are as {!contract}
    states.

    Raises [Invalid_argument] if [p] has one operand, more than 11, or other
    than [n]; the dtype is a boolean; an operand's rank is not the number of
    axes [p] gives it; [...] covers different numbers of axes in two operands;
    a name's extents differ between operands; every tree holds a value of
    more than {!Nx_array.Layout.max_rank} axes; a group's extents do not
    multiply to its axis, or it leaves two extents unknown; a unit axis's
    extent is not [1]; a name in [sizes] is not in [p], or its size is
    negative; or [acc] is a boolean, a float narrower than [float32] or than
    the operands, or of another kind than theirs. *)

val matmul : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [matmul a b] is the matrix product over the last two axes, leading axes
    broadcast, at [a]'s dtype and accumulated as {!contract} does by default. A
    1-d [a] is a row and a 1-d [b] a column, and their unit axis is dropped from
    the result: two 1-d operands give their 0-d inner product.

    Raises [Invalid_argument] if an operand is 0-d, the inner extents differ,
    the leading axes do not broadcast, or the dtype is a boolean. *)

val dot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [dot a b] sums the products of [a]'s last axis with [b]'s only axis, for a
    1-d [b], or its second to last otherwise. The result's axes are [a]'s
    others, then [b]'s: two vectors give their 0-d inner product, two matrices
    their product. It does not broadcast, and accumulates as {!matmul}.

    Raises [Invalid_argument] if an operand is 0-d, the summed extents differ,
    or the dtype is a boolean. *)

val vdot : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [vdot a b] is the 0-d sum of the products of [conjugate a] and [b], both
    flattened, accumulated as {!matmul}.

    Raises [Invalid_argument] if [a] and [b] have different numbers of
    elements, or the dtype is a boolean. *)

val conv :
  ?stride:int array ->
  ?padding:(int * int) array ->
  ?dilation:int array ->
  ?groups:int ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [conv x w] slides the kernel [w], [[o; k…; c / groups]], over [x],
    [[n; s…; c]], channels last, for [K] spatial axes [s…] and kernel extents
    [k…]: the result, [[n; t…; o]], is at [(b, p…, j)] the sum over [i] and
    the taps [q…] of [x[b, p · stride + q · dilation - lo…, i]] times
    [w[j, q…, i]], an element outside [x] reading zero. The kernel is not
    flipped. [x] is padded along each spatial axis by [padding]'s [(lo, hi)]
    (default none), a negative one cutting; [stride] and [dilation] default
    to ones; and [t = (s + lo + hi - dilation (k - 1) - 1) / stride + 1], none
    where the padded axis is shorter than the dilated kernel. Under [groups]
    (default [1]) the channels split into [groups] runs, output channels
    [g · o / groups] to [(g + 1) · o / groups - 1] reading input run [g]
    alone.

    It is one reduction over the padded windows of [x], which are never
    stored, accumulated as {!contract} does by default and rounded once:
    the float formats narrower than 32 bits sum at [float32]. Integers wrap.

    Raises [Invalid_argument] if [x] has fewer than three axes; [w] has
    another rank; [stride], [padding] or [dilation] do not have [K] entries;
    a stride or a dilation is below [1]; a kernel extent is [0]; [groups] is
    below [1] or does not divide [o]; [x]'s channels are not [groups] times
    [w]'s last extent; or the dtype is a boolean. *)

val conv_transpose :
  ?stride:int array ->
  ?padding:(int * int) array ->
  ?dilation:int array ->
  ?groups:int ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [conv_transpose x w] is {!conv}'s transpose with respect to its input:
    for the [z] of the least spatial extents whose [conv z w], under the same
    arguments, has [x]'s shape [[n; t…; o]], it is the [[n; s…; c]] value
    with [sum (mul (conv z w) x) = sum (mul z (conv_transpose x w))] for
    every such [z]: [s = (t - 1) stride + dilation (k - 1) + 1 - lo - hi], or
    [0] where that is negative or [t = 0]. It is one reduction over [x] with
    [stride - 1] zeros between its elements, windowed by [w] reversed along
    its spatial axes, never stored.

    Raises [Invalid_argument] as {!conv}, reading [o] from [x]'s last axis, or
    if no input has a [conv] of [x]'s shape. *)

(** {2:matrices Matrices}

    A value of two axes or more is a batch of matrices over its last two:
    rows, then columns. *)

val matrix_transpose : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [matrix_transpose x] is [x] with its last two axes exchanged; [x] itself
    below two axes. A view. *)

val tril : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [tril ~k x] is [x] where column - row [<= k] (default [0]) and zero above,
    in each matrix of the batch.

    Raises [Invalid_argument] if [x] has fewer than two axes. *)

val triu : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [triu ~k x] is [x] where column - row [>= k] (default [0]) and zero below,
    as {!tril}. *)

val diagonal :
  ?offset:int -> ?axis1:int -> ?axis2:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [diagonal ~offset ~axis1 ~axis2 x] is the elements of [x] at index [i]
    along [axis1] and [i + offset] along [axis2] (default the last two axes,
    and [0]), for each [i] where both lie in their axes; that axis replaces the
    two, last, the others kept in order. A view where [x]'s two axes merge
    without a copy.

    Raises [Invalid_argument] if [x] has fewer than two axes, or [axis1] and
    [axis2] are not two distinct axes of [x]. *)

val trace : ?offset:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [trace ~offset x] is the {!sum} of [diagonal ~offset x] along its last
    axis: one value per matrix of the batch, [+0] for an empty diagonal.

    Raises [Invalid_argument] as {!diagonal} and {!sum} do. *)

val diag : ?k:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [diag ~k v] is, for a vector [v] of [n] elements, the square of
    [n + |k|] whose [k]th diagonal (default [0], above the main one where
    positive) holds [v] and whose other elements are zero; for a matrix, its
    diagonal [diagonal ~offset:k v].

    Raises [Invalid_argument] if [v] has neither one nor two axes. *)

(** {1:linalg Linear algebra}

    Factorisations, solves and norms of the matrices of a batch, laid out as
    {!section-matrices} says, [m] rows, [n] columns and [k = min m n]. They
    compute [float32], [float64], [complex64] and [complex128]: another dtype
    raises [Invalid_argument]. [aᴴ] is the conjugate transpose, the transpose
    of a real [a].

    Every result has its operand's dtype. A real result of a complex operand
    (a singular value, an eigenvalue of {!eigh}, {!slogdet}'s logarithm, a
    norm) is complex with an imaginary part of [+0], and {!real} extracts it.

    Each matrix of a batch is computed on its own, and data never raises: a
    matrix whose factorisation fails, as each function states, is NaN in its
    own results alone. For a finite [a],
    [all ~axes:[ -2; -1 ] (isfinite (cholesky a))] tests which matrices of a
    batch are positive definite.

    Each kernel library states its error bounds. A factorisation leaves some
    choices open: the order of {!eig}'s eigenvalues; {!qr}'s factors, unique
    up to a unit factor per column of [q] and row of [r] where [r]'s diagonal
    has no zero; the vectors of {!svd}, {!eigh} and {!eig}, unique up to a
    unit factor where their value is simple and up to a unitary basis of its
    space where it repeats; and the columns [Complete] factors add. *)

(** The type for a square matrix's triangles, each with its diagonal: the
    elements on and below it, or on and above it. *)
type triangle = Lower | Upper

(** The type for how much of a matrix's unitary factors a factorisation gives:
    [k] columns of the left one and [k] rows of the right one, or the whole
    squares, of [m] and of [n]. *)
type factors = Reduced | Complete

val cholesky : ?triangle:triangle -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [cholesky a] is the factor of the Hermitian positive definite matrix that
    [a]'s [triangle] (default [Lower]) and the real parts of its diagonal name,
    which alone are read: [L], lower triangular with a real and positive
    diagonal and [a = L Lᴴ], or under [~triangle:Upper] [U = Lᴴ], upper
    triangular with [a = Uᴴ U]. The other triangle is [+0]. A matrix whose
    factorisation meets a pivot that is not positive, or NaN, is NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val qr : ?factors:factors -> ('v, 's, 'd) t -> ('v, 's, 'd) t * ('v, 's, 'd) t
(** [qr a] is [(q, r)] with [a = q r], [q]'s columns orthonormal and [r] upper
    triangular: [q] is [[…; m; k]] and [r] [[…; k; n]], or under
    [~factors:Complete] [[…; m; m]] and [[…; m; n]].

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val lu : ('v, 's, 'd) t -> 'd int64_t * ('v, 's, 'd) t * ('v, 's, 'd) t
(** [lu a] is [(perm, l, u)] where row [i] of [matmul l u] is row [perm.{i}] of
    [a], [l] is [[…; m; k]], lower triangular with ones on its diagonal, and [u]
    is [[…; k; n]], upper triangular; [perm] is [[…; m]].

    The factorisation pivots partially: at each column it moves up the row at
    or below the diagonal whose magnitude ([|re| + |im|] for a complex) is
    greatest, the first such row on a tie, so each element of a real [l] has a
    magnitude of at most [1]. A NaN is never greater, and nothing is greater
    than a NaN. A singular [a] factors as any other, its zero pivots on [u]'s
    diagonal.

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val svd :
  ?vectors:bool ->
  ?factors:factors ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * ('v, 's, 'd) t * ('v, 's, 'd) t
(** [svd a] is [(u, s, vh)] with [a = matmul (mul u (unsqueeze ~axes:[ -2 ] s))
    vh], [s] the [[…; k]] singular values, descending and not negative, a zero
    one [+0]; [u] and [vh] have orthonormal columns and rows: [[…; m; k]] and
    [[…; k; n]], or under [~factors:Complete] [[…; m; m]] and [[…; n; n]].
    Under [~vectors:false] (default [true]) neither [u] nor [vh] is computed:
    they are [[…; m; 0]] and [[…; 0; n]]. A matrix that holds NaN or an
    infinity, or on which the iteration does not converge, is NaN in all three.

    Raises [Invalid_argument] if [a] has fewer than two axes, or for its
    dtype. *)

val eig :
  ?vectors:bool ->
  (Complex.t, 's, 'd) t ->
  (Complex.t, 's, 'd) t * (Complex.t, 's, 'd) t
(** [eig a] is [(w, v)], the [[…; n]] eigenvalues of [a] and columns of unit
    norm [v] with [matmul a v = mul v (unsqueeze ~axes:[ -2 ] w)]. A real
    matrix is cast first: [eig (cast complex128 a)]. Under [~vectors:false]
    (default [true]) [v] is not computed and is [[…; n; 0]]. NaN as for
    {!svd}.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not
    square. *)

val eigh :
  ?vectors:bool ->
  ?triangle:triangle ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * ('v, 's, 'd) t
(** [eigh a] is [(w, v)], the [[…; n]] eigenvalues of the Hermitian [a],
    ascending, and orthonormal columns [v] with
    [matmul a v = mul v (unsqueeze ~axes:[ -2 ] w)]. [a] is the Hermitian
    matrix its [triangle] (default [Lower]) and the real parts of its diagonal
    name, which alone are read. Under [~vectors:false] (default [true]) [v] is
    not computed and is [[…; n; 0]]. A matrix whose read part holds NaN or an
    infinity, or on which the iteration does not converge, is NaN in both.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val det : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [det a] is the determinant of each matrix of [a], the product of the
    diagonal of {!lu}'s [u] negated once per row interchange. A zero
    determinant is [+0], and a matrix with no row has determinant [1]. The
    product is taken in the dtype and overflows or underflows where a partial
    product does. {!slogdet} holds determinants beyond the dtype's range.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val slogdet : ('v, 's, 'd) t -> ('v, 's, 'd) t * ('v, 's, 'd) t
(** [slogdet a] is [(sign, logabs)] with [det a = sign · exp logabs], from
    {!lu}'s pivots: [logabs] is the sum of their magnitudes' logarithms, and
    [sign] is [1.] or [-1.] for a real [a] and a complex number of magnitude
    [1.] for a complex one. A singular matrix, one with a zero pivot, has
    [sign] [+0] and [logabs] [neg_infinity] whatever its other pivots.

    Raises [Invalid_argument] as {!det}. *)

(** The type for the matrix a solve applies: [aᵀ], or [aᴴ]. For a real [a]
    they are one. *)
type transpose = Transpose | Adjoint

val solve_triangular :
  ?triangle:triangle ->
  ?transpose:transpose ->
  ?unit_diag:bool ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [solve_triangular a b] is [x] with [matmul a x = b], or under [transpose]
    with [matmul aᵀ x = b] or [matmul aᴴ x = b], reading [a]'s [triangle]
    (default [Lower]) alone, without its diagonal under [~unit_diag:true]
    (default [false]), whose ones it takes instead. A [b]
    with one axis fewer than [a] is a vector, [[…; n]], and [x] has its shape;
    otherwise [b] is [[…; n; r]], [r] right-hand sides. [a]'s and [b]'s
    leading axes broadcast. A zero on a diagonal it reads makes that matrix's
    [x] NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    for its dtype, if [b] has neither [a]'s rank nor one axis fewer, if [b]'s
    rows are not [n], or if the leading axes do not broadcast. *)

val solve : ('v, 's, 'd) t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [solve a b] is [x] with [matmul a x = b], for a square [a], from {!lu}'s
    factors and two triangular solves; [b] is as {!solve_triangular} takes
    it. A matrix with a pivot of exactly zero makes its [x] NaN. A matrix
    singular in exact arithmetic whose pivot rounds to a tiny nonzero value
    gives large, infinite or NaN elements instead.

    Raises [Invalid_argument] as {!solve_triangular}. *)

val inv : ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [inv a] is [solve a] of the identity: NaN for a matrix {!solve} finds
    singular.

    Raises [Invalid_argument] if [a] has fewer than two axes or is not square,
    or for its dtype. *)

val lstsq :
  ?rtol:float ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t * ('v, 's, 'd) t * 'd int64_t * ('v, 's, 'd) t
(** [lstsq a b] is [(x, residuals, rank, s)]: [x] the least-squares solution
    of [matmul a x ≈ b] of least norm, whatever [a]'s shape and rank, from
    [s], {!svd}'s singular values of [a]. A singular value at or below [rtol]
    times [max m n] times its matrix's greatest counts as zero; [rtol]
    defaults to the dtype's machine epsilon. [rank] is the number of the
    others, [-1] for a matrix on which {!svd} fails, whose [x] is NaN.
    [residuals] is the squared norm of each column of [b - matmul a x]: [b]'s
    shape without its rows. [b] is as {!solve_triangular} takes it, with [m]
    rows.

    Raises [Invalid_argument] if [a] has fewer than two axes, for its dtype,
    if [b] has neither [a]'s rank nor one axis fewer, if [b]'s rows are not
    [m], if the leading axes do not broadcast, or if [rtol] is negative or
    NaN. *)

val pinv : ?rtol:float -> ?hermitian:bool -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [pinv a] is the [[…; n; m]] Moore-Penrose pseudoinverse of [a], from
    {!svd}, or from {!eigh} of the Hermitian [a] its lower triangle names under
    [~hermitian:true] (default [false]). A singular value, or an eigenvalue's
    magnitude, at or below [rtol] times [max m n] times its matrix's greatest
    counts as zero; [rtol] defaults to the dtype's machine epsilon. A matrix on
    which that factorisation fails is NaN.

    Raises [Invalid_argument] if [a] has fewer than two axes, is not square
    under [~hermitian:true], for its dtype, or if [rtol] is negative or NaN. *)

val vector_norm :
  ?ord:float ->
  ?axes:int list ->
  ?keepdims:bool ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [vector_norm x] is the norm of order [ord] (default [2.]) of the vector of
    the elements of [x] along [axes] (default all of them), dropping those
    axes unless [keepdims]. Of the magnitudes [|x|], it is
    [(Σ |x|^ord)^(1/ord)] for a nonzero [ord], the count of nonzero elements
    for [0.], and the greatest or least magnitude for [infinity] or
    [neg_infinity]. A negative [ord] over no element is [infinity].

    The sums of powers are scaled by the greatest magnitude, or the least for
    a negative [ord], so that a norm the dtype holds never overflows or
    underflows on the way: the norm of [[1e200; 1e200]] is [1.414e200]. A
    NaN element makes its norm NaN.

    Raises [Invalid_argument] for [x]'s dtype, if an axis is outside [x]'s
    rank or repeats, if [ord] is NaN, or if [ord] is [infinity] or
    [neg_infinity] and [axes] hold no element. *)

val matrix_norm :
  ?ord:[ `Fro | `Nuc | `One | `Two | `Inf | `NegOne | `NegTwo | `NegInf ] ->
  ?keepdims:bool ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [matrix_norm x] is the norm of order [ord] (default [`Fro]) of each matrix
    of [x], dropping its last two axes unless [keepdims]. Of a matrix:
    - [`Fro] is [sqrt (Σ |x|²)], scaled as {!vector_norm}'s, and [`Nuc] the
      sum of its singular values.
    - [`Two] is its greatest singular value and [`NegTwo] its least.
    - [`One] is the greatest sum of a column's magnitudes and [`NegOne] the
      least; [`Inf] and [`NegInf] are the same of its rows.

    Raises [Invalid_argument] if [x] has fewer than two axes, for its dtype,
    or if [ord] takes a greatest or least singular value, column or row of a
    matrix that has none. *)

(** {1:fft Fourier transforms}

    Along an axis of [n] points, the forward transform is
    [y[k] = Σ_j x[j] e^(-2πi jk/n)] and the inverse the same sum with
    [e^(+2πi jk/n)]. A transform along several axes is the transform along
    each in turn. A transform of no point is the empty sum, zero.

    [axes] defaults to the last axis. [s] gives the number of points along each
    of [axes], in order: the operand is cut to it, or extended with zeros. A
    transform of [N] points in all is scaled by [norm]: under [Backward]
    (default) the inverse divides by [N], under [Forward] the forward one does,
    and under [Ortho] each divides by [√N].

    Each raises [Invalid_argument] if the operand is 0-d, if [axes] is empty,
    if an axis is outside the operand's rank or repeats, if [s] does not give
    one length per axis, or if a length is negative. *)

(** The type for the scaling of a transform and its inverse. *)
type fft_norm = Backward | Forward | Ortho

val fft :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (Complex.t, 's, 'd) t
(** [fft x] is the forward transform of [x] along [axes]. *)

val ifft :
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (Complex.t, 's, 'd) t
(** [ifft x] is the inverse transform of [x] along [axes]: under one [norm],
    [ifft (fft x)] is [x] to rounding. *)

val rfft :
  (Complex.t, 'c) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [rfft dt x] is the forward transform of the real [x] along [axes], the
    last of them keeping bins [0] to [n/2] of its [n] points, the others
    being redundant: one bin for [n = 0]. It computes at the wider of [x]'s
    precision and [dt]'s, the formats narrower than 32 bits at [float32], and
    rounds the scaled result once to [dt]. *)

val irfft :
  (float, 'r) dtype ->
  ?axes:int list ->
  ?s:int list ->
  ?norm:fft_norm ->
  (Complex.t, 's, 'd) t ->
  (float, 'r, 'd) t
(** [irfft dt x] is the real inverse of {!rfft} along [axes], of [n] points
    along the last of them from its first [n/2 + 1] bins, which are cut or
    extended with zeros to that many. [s] gives the points of the result;
    without it, the other axes keep their extents and [n] is [2 (m - 1)] for
    [m] bins. Along the last axis, bin [k] above [n/2] reads as the conjugate
    of bin [n - k], and the imaginary parts of bin [0] and, for an even [n],
    of bin [n/2] are not read: it is this linear map for every operand,
    Hermitian or not. It computes and rounds as {!rfft}.

    Raises [Invalid_argument] also if the last axis has no bin and [s] does
    not give its points. *)

(** The type for the four discrete cosine and sine transforms, I to IV. *)
type trig_type = Type_1 | Type_2 | Type_3 | Type_4

val dct :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [dct x] is the discrete cosine transform of type [type_] (default
    [Type_2]) of [x] along each of [axes]: of [n] points [x[j]], type II is
    [y[k] = 2 Σ_j x[j] cos (πk(2j + 1) / 2n)], type I
    [y[k] = x[0] + (-1)^k x[n-1] + 2 Σ_(0<j<n-1) x[j] cos (πjk / (n-1))],
    type III [y[k] = x[0] + 2 Σ_(j>0) x[j] cos (πj(2k + 1) / 2n)] and type IV
    [y[k] = 2 Σ_j x[j] cos (π(2j + 1)(2k + 1) / 4n)]. Under [Forward] it
    divides by the transform's logical length, [2 (n - 1)] for type I and
    [2n] otherwise, and under [Ortho] it is the orthonormal transform. A
    copy of [x]'s shape and dtype. It computes the formats narrower than 32
    bits at [float32] and rounds the result once.

    Raises [Invalid_argument] also if type I is asked of an axis of one
    element. *)

val idct :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [idct x] is the inverse of {!dct} under the same [type_] and [norm]: types
    I and IV are their own inverses up to scale, II and III each other's.

    Raises [Invalid_argument] as {!dct}. *)

val dst :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [dst x] is the discrete sine transform of type [type_] (default [Type_2])
    along each of [axes]: type II is
    [y[k] = 2 Σ_j x[j] sin (π(k + 1)(2j + 1) / 2n)], type I
    [y[k] = 2 Σ_j x[j] sin (π(j + 1)(k + 1) / (n + 1))], type III
    [y[k] = (-1)^k x[n-1] + 2 Σ_(j<n-1) x[j] sin (π(j + 1)(2k + 1) / 2n)] and
    type IV [y[k] = 2 Σ_j x[j] sin (π(2j + 1)(2k + 1) / 4n)]. Its logical
    length is [2 (n + 1)] for type I and [2n] otherwise, and [norm] scales and
    narrow formats compute as {!dct}'s. *)

val idst :
  ?type_:trig_type ->
  ?axes:int list ->
  ?norm:fft_norm ->
  (float, 's, 'd) t ->
  (float, 's, 'd) t
(** [idst x] is the inverse of {!dst} under the same [type_] and [norm]. *)

val hann : (float, 's) dtype -> int -> (float, 's, 'd) t
(** [hann dt n] is the periodic Hann window of [n] points,
    [0.5 - 0.5 cos (2πi / n)], computed at float64 and rounded once: the point
    that would open the next period is dropped, so shifted copies sum to a
    constant. A value of every set.

    Raises [Invalid_argument] if [n] is negative. *)

val stft :
  (Complex.t, 'c) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 's, 'd) t ->
  (float, 's, 'd) t ->
  (Complex.t, 'c, 'd) t
(** [stft dt ~window x] is the short-time transform of [x] along its last
    axis: frames of [window] samples, one every [step] (default [window / 4],
    at least [1]), each multiplied by [win] and transformed by {!rfft}.
    [[…; n]] gives [[…; frames; window / 2 + 1]], time first, with
    [frames = (n - window) / step + 1]; samples past the last whole frame are
    dropped. [win] defaults to [hann (dtype x) window], and
    [ones (dtype x) [| window |]] frames without a taper.

    Raises [Invalid_argument] if [x] is 0-d, [window] or [step] is below [1],
    [window] exceeds the last axis, or [win] is not [[window]]. *)

val istft :
  (float, 'r) dtype ->
  window:int ->
  ?step:int ->
  ?win:(float, 'r, 'd) t ->
  ?length:int ->
  (Complex.t, 'c, 'd) t ->
  (float, 'r, 'd) t
(** [istft dt ~window z] is the signal whose {!stft} under the same [window],
    [step] and [win] is [z]: each frame inverted by {!irfft}, multiplied by
    [win], summed back where it was taken, and divided by the sum of [win]'s
    squares landing there. [[…; frames; window / 2 + 1]] gives
    [[…; (frames - 1) step + window]], or [length] samples, cut or extended
    with zeros. A sample every frame's [win] holds at zero carries nothing and
    is [0.]: under the default taper, sample [0].

    Raises [Invalid_argument] if [z] has fewer than two axes, [window] is below
    [1], [step] is outside [[1, window]], the last axis is not
    [window / 2 + 1], [win] is not [[window]], or [length < 1]. *)

(** {1:order Order and counted lengths}

    Sorting, selection, searching and grouping order elements by one order, the
    {e sort order}:
    - floats as [-∞ < … < -0. < +0. < … < +∞ < NaN], every NaN equal to every
      other whatever its sign and payload;
    - [false < true], and integers by value, unsigned ones as unsigned;
    - complex numbers by real part, then imaginary part, each as a float; one
      with a NaN part equals every other such and is above every other complex
      number.

    A descending sort orders by its exact reverse: NaN first, [+0.] before
    [-0.]. Keys are elements, or rows of a 2-d value compared column [0] first:
    {!lexsort} sorts them, {!searchsorted} searches them and {!unique} groups
    them. Complex numbers have no keys. *)

val sort :
  ?descending:bool -> ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t * 'd int64_t
(** [sort ~descending ~axis x] is [x]'s elements along [axis] (default [-1]) in
    the sort order, or its reverse where [descending] (default [false]), and the
    positions they come from. The sort is stable in both directions: equal
    elements keep their order, NaNs included. The values are
    [take_along_axis ~axis positions x], bit for bit.

    Raises [Invalid_argument] if [axis] is not an axis of [x]. *)

val argsort : ?descending:bool -> ?axis:int -> ('v, 's, 'd) t -> 'd int64_t
(** [argsort ~descending ~axis x] is [snd (sort ~descending ~axis x)]. *)

val top_k : k:int -> ?axis:int -> ('v, 's, 'd) t -> ('v, 's, 'd) t * 'd int64_t
(** [top_k ~k ~axis x] is the first [k] entries along [axis] (default [-1]) of
    [sort ~descending:true ~axis x]: the greatest first, equal ones lowest
    position first, a NaN before every number. One sort that keeps [k].

    Raises [Invalid_argument] if [axis] is not an axis of [x] or [k] is outside
    [[0, n]], [n] the axis' extent. *)

val order_key : ('k, 'r) dtype -> ('v, 's, 'd) t -> ('k, 'r, 'd) t
(** [order_key kd x] is, for each element of [x], an unsigned integer of [kd]
    whose order is the sort order: two keys compare, read unsigned, as their
    elements sort, and are equal iff the elements are equal in the sort order.
    [kd] is {!uint8}, {!uint16}, {!uint32} or {!uint64}, at least as wide as
    [x]'s elements; [uint64] keys every dtype but the complex ones.

    The key of an element is:
    - for a signed integer, its value in the signed integer of [kd]'s width with
      the sign bit flipped;
    - for an unsigned integer, its value, and for a boolean [0] or [1];
    - for a float, the bits of its value in the float of [kd]'s width, all of
      them flipped where the sign bit is set and only the sign bit elsewhere, so
      [-0.] keys just below [+0.]; every NaN keys to all ones. The float of a
      width is [x]'s own dtype at its width, bfloat16 for bfloat16 at 16 bits,
      float8_e4m3fn for float4_e2m1fn at 8 bits, and otherwise float16, float32
      or float64.

    An exact cast to a dtype no wider than [kd] keeps the key wherever the two
    dtypes key through the same float: [order_key uint64 (cast int64 x)] is
    [order_key uint64 x] for every signed integer [x]. A key of several columns
    is a row of order keys made with {!stack} [~axis:1]; a column that sorts
    descending is {!bitwise_not} of its key.

    Raises [Invalid_argument] if [x] is complex, or if [kd] is not one of the
    four or is narrower than [x]'s elements. *)

val lexsort : ('v, 's, 'd) t -> 'd int64_t
(** [lexsort keys] is the stable permutation that sorts the keys of [keys]: [n]
    elements of a 1-d [keys], or [n] rows of a 2-d one, compared column [0]
    first, each in the sort order. [take ~axis:0 (lexsort keys) keys] is in
    order and equal keys keep their order; a 2-d [keys] with no column leaves
    every row in place. It sorts one column at a time, the last first: [w] sorts
    for [w] columns.

    Raises [Invalid_argument] if [keys] is neither 1-d nor 2-d, or is complex.
*)

val searchsorted :
  side:[ `Left | `Right ] -> ('v, 's, 'd) t -> ('v, 's, 'd) t -> 'd int64_t
(** [searchsorted ~side s v] is, for each key of [v], the number of keys of [s]
    before it ([`Left]), or before or equal to it ([`Right]). Keys compare in
    the sort order, except that [-0.] equals [+0.], as {!less} compares numbers.
    - A 1-d [s] holds [m] keys of one element; [v] holds keys of one element in
      any shape, and the result has [v]'s shape.
    - A 2-d [s] of shape [[|m; w|]] holds [m] rows; [v] has shape [[|n; w|]],
      and the result shape [[|n|]].

    It is a binary search of [⌈log₂ (m + 1)⌉] rounds, each a gather of one key
    of [s] per key of [v]. Where [s] is not in order, each result is still in
    [[0, m]].

    Raises [Invalid_argument] if [s] is neither 1-d nor 2-d, is complex, or is
    2-d and [v] is not 2-d with as many columns. *)

(** {2:counted Lengths that are data}

    A length that is data is read once, on the host, as the function's one read:
    a placed operand's set synchronises once. A traced value cannot be read, so
    there the function raises naming itself, unless [size] fixes the length, in
    which case nothing is read. *)

val positions : ?size:int -> ('v, 's, 'd) t -> 'd int64_t
(** [positions c] is each index [i] of the 1-d [c] repeated [c.{i}] times, in
    increasing order. A boolean counts [1] where it holds, so [positions c] is
    the indices where [c] holds; a negative count counts zero, as does a
    [uint64] count past [int64]'s range. Without [size], the result's length is
    the counts' sum, read on the host. With [size], the result has [size]
    entries and reads nothing: the first positions, then [dim 0 c]. Along an
    axis of [c]'s length that pad is out of range, so {!take} reads zero there
    and {!scatter} writes nothing; along a longer axis it is a real position.

    Raises [Invalid_argument] if [c] is not 1-d, is a float or a complex, or if
    [size] is negative. *)

type 'd groups = {
  ids : 'd int64_t;  (** Each key's group, [n] entries. *)
  first : 'd int64_t;  (** Each group's first key, ascending. *)
  counts : 'd int64_t;  (** Each group's number of keys. *)
  count : 'd int64_t;  (** [k], 0-d. *)
}
(** The type for [n] keys in [k] groups of equal keys, numbered in order of
    first appearance. *)

val unique : ?size:int -> ('v, 's, 'd) t -> 'd groups
(** [unique keys] groups the keys of [keys] (elements of a 1-d [keys], rows of a
    2-d one) that are equal in the sort order: every NaN is one group, [-0.] and
    [+0.] two. For [g = unique keys], [g.ids.{0}] is [0],
    [take ~axis:0 g.first keys] is the distinct keys in order of first
    appearance, and key [i]'s group starts at [g.first.{g.ids.{i}}], at most
    [i].

    Without [size], [first] and [counts] have [k] entries, [k] read on the host.
    With [size] they have [size] entries and nothing is read: past [k], [first]
    holds [n] and [counts] [0]; below [k], the later groups are cut from them,
    and [ids] still numbers those groups from [size].

    It is one stable sort of the keys and a few passes over them.

    Raises [Invalid_argument] if [keys] is neither 1-d nor 2-d, is complex, or
    if [size] is negative. *)

val quantile :
  ?axis:int -> float array -> (float, 's, 'd) t -> (float, 's, 'd) t
(** [quantile ~axis qs x] is the quantiles [qs] of [x] along [axis], or of all
    its elements without [axis]. The result leads with an axis of one entry per
    probability, followed by [x]'s axes without [axis].

    The quantile [q] of [n] elements interpolates linearly between the order
    statistics around [h = q (n - 1)] in the sort order: with [a] and [b] the
    elements at [floor h] and [floor h + 1], it is [a + (h - floor h) (b - a)],
    and [a] exactly where [h] is an integer or [a] equals [b]. NaN sorts last,
    so it reaches the upper quantiles. The narrow floats interpolate in float32
    and round once. One sort serves every probability.

    Raises [Invalid_argument] if a probability is outside [[0, 1]] or NaN, if
    [axis] is not an axis of [x], or if there is no element along it. *)

val histogram :
  ?weights:(float, 's, 'd) t ->
  ((float, 's, 'd) t * (float, 's, 'd) t) list ->
  (float, 's, 'd) t
(** [histogram [ (e1, x1); …; (ed, xd) ]] counts points into the cells of the
    edges [e1] to [ed]: point [i] is [(x1.{i}, …, xd.{i})], one per element of
    the coordinates, which share one shape. The result has one axis per
    dimension, of [dim 0 ek - 1] bins; cell [(j1, …, jd)] counts the points
    whose coordinate [xk] is in bin [jk] of [ek] for every [k].
    - Bin [j] holds the coordinates from [ek.{j}] included to [ek.{j+1}]
      excluded; the last bin includes its right edge too. Coordinates compare as
      {!less} does, so [-0.] is [+0.].
    - A point outside the edges along a dimension, NaN included, is in no cell.
    - Counts accumulate in [int64] and convert once to the dtype. With
      [weights], of the coordinates' shape, a cell holds the sum of its points'
      weights, as {!scatter}'s [Add] sums them.

    Edges must ascend; along edges that do not, a point lands in some cell or in
    none.

    Raises [Invalid_argument] if no dimension is given, if an edge vector is not
    1-d of two or more edges, if the coordinates' shapes differ, or if [weights]
    has another shape. *)

(** {1:windows Windows and segments}

    A function here reads its operand in groups of elements: classes, windows
    along its last axes, or rows grouped by positions held in data. Each
    computes where its operands lie.

    The {e spatial} axes are an operand's last [K] axes, and the axes before
    them are batch axes, each computed alone. Along a spatial axis of [n]
    elements, padded with [lo] zeros before and [hi] after, a window of [k]
    elements with stride [s] and dilation [d] holds the padded axis' elements
    [w s + j d], [j < k], for window [w]. There are
    [(n + lo + hi - d (k - 1) - 1) / s + 1] windows, and none where
    [d (k - 1) + 1] exceeds [n + lo + hi]. Windows across several spatial axes
    are every combination of one window per axis, in C order, and so are
    their taps [j]. Except for {!extract_patches}, whose result they are,
    these functions read the windows where they lie and store none. *)

val one_hot : num_classes:int -> ('v, 's, 'd) t -> 'd uint8_t
(** [one_hot ~num_classes i] is [i]'s shape with an axis of [num_classes]
    appended, holding [1] at the class each element of [i] names and [0] at the
    others: an element outside [\[0, num_classes)] gives a row of zeros.
    Integers only.

    Raises [Invalid_argument] if [i] is not of an integer dtype or
    [num_classes] is negative. *)

val extract_patches :
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [extract_patches ~kernel_size ~stride ~dilation ~padding x] is the windows
    of [x] over its last [K = Array.length kernel_size] axes, padded with zeros
    by [padding]'s [(lo, hi)] pairs. A value of shape [[lead…; spatial…]]
    gives [[lead…; P; L]], [P] the product of [kernel_size] and [L] the number
    of windows: its element at [(…, p, l)] is window [l]'s tap [p]. A copy.

    Raises [Invalid_argument] if [kernel_size] is empty; [stride], [dilation]
    and [padding] do not have one entry per kernel axis; a size, stride or
    dilation is below [1] or a padding negative; or [x] has fewer axes than
    [kernel_size]. *)

val combine_patches :
  output_size:int array ->
  kernel_size:int array ->
  stride:int array ->
  dilation:int array ->
  padding:(int * int) array ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding p]
    is the adjoint of {!extract_patches} from spatial axes [output_size]: [p],
    of shape [[lead…; P; L]], is summed into [[lead…; output_size…]], each of
    its elements added at the position that {!extract_patches} reads it from,
    and one that lands in the padding dropped. An element no window covers is
    [+0]. With one geometry, the sum of [extract_patches x * y] is the sum of
    [x * combine_patches y].

    The sums add in C order of the taps, from [+0], and the float formats
    narrower than 32 bits sum at [float32] and round once. Integers wrap.
    Every dtype but booleans.

    Raises [Invalid_argument] as {!extract_patches} does on the geometry, if
    [output_size] does not have one entry per kernel axis or has a negative
    one, if [p] has fewer than two axes or its last two are not [P] and [L],
    or if [p] is boolean. *)

val correlate :
  ?padding:[ `Full | `Same | `Valid ] ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [correlate ~padding x kernel] is the cross-correlation of [x] with
    [kernel] over the last [K = ndim kernel] axes of [x], each sum of products
    taken without flipping [kernel]. [padding] defaults to [`Valid].

    Along each spatial axis, of [n] elements in [x] and [k] in [kernel],
    value [i] correlates [kernel] with the elements of [x] from [i - k + 1],
    those outside [x] being zero. [padding] keeps a run of consecutive values.
    With [m = min n k]:
    - [`Full] keeps the [n + k - 1] from value [0];
    - [`Valid] keeps the [max n k - m + 1] where the shorter of [x] and
      [kernel] lies within the longer, from value [m - 1]. An empty [x] lies
      within [kernel] at [k + 1] positions, so it gives [k + 1] zeros;
    - [`Same] keeps [max n k], from value [(m - 1) / 2] if [k <= n] and from
      value [m / 2] if [k > n]. With [k <= n], value [i] correlates [kernel]
      with the elements of [x] from [i - k / 2] to [i + (k - 1) / 2]: an odd
      kernel is centred on element [i].

    For [x = [|1.; 2.; 3.; 4.; 5.; 6.; 7.|]] and
    [kernel = [|1.; 10.; 100.; 1000.|]], [`Same] gives
    [[|2100.; 3210.; 4321.; 5432.; 6543.; 7654.; 765.|]].

    Each value sums one product per element of [kernel], from [+0], in an
    order fixed by the shapes, as {!sum} adds. The float formats narrower
    than 32 bits multiply and sum at [float32] and round once. Integers wrap.
    Every dtype but booleans.

    Raises [Invalid_argument] if [x] has fewer axes than [kernel], [kernel] has
    an axis of no element, or the dtype is boolean. *)

(** {2:segments Segments and ranges}

    A {e row} of [x] is its value at one position of its first axis. A
    {e segment} is the rows whose id, held in data, is one number; a
    {e range} is the rows between two bounds held in data. As positions held
    in data, an id or a bound never raises: an id outside the segments drops
    its row, and a bound clips into the rows.

    {!reduce_segments} and {!reduce_ranges} fold each segment's or range's
    rows, in row order, by a {!combine}:
    - [Add] sums from [+0], so a sum of [-0.] rows is [+0.]; the float formats
      narrower than 32 bits sum at [float32] and round once, and integers
      wrap. Every dtype but booleans.
    - [Max] and [Min] keep the greatest or least row as {!maximum} orders
      them, or the first NaN, as {!max} does. Every dtype.
    - [Set] keeps the last row.

    A segment or range of no row holds [Add]'s and [Set]'s zero, or [Max]'s
    least and [Min]'s greatest value in the dtype's order: [neg_infinity] and
    [infinity], [-448.] and [448.] in [float8_e4m3fn], which has no infinity,
    the least and greatest integers, [false] and [true], and for complex
    numbers both parts so. *)

val reduce_segments :
  combine -> segments:int -> 'd int64_t -> ('v, 's, 'd) t -> ('v, 's, 'd) t
(** [reduce_segments c ~segments ids x] is [segments] rows, row [s] the rows [i]
    of [x] with [ids.{i} = s] folded by [c]. [ids] holds one id per row: give
    a row that belongs to no segment the id [-1]. It is
    [scatter ~combine:c ~axis:0] of [x]'s rows into [segments] rows of the
    identity, so its sums have {!scatter}'s bits.

    {[
    let ids = Nx.create Nx.int64 [| 5 |] [| 0L; 2L; 0L; -1L; 2L |]
    let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    (* [|4.; 0.; 7.|] and [|3.; neg_infinity; 5.|] *)
    let sums = Nx.reduce_segments Add ~segments:3 ids x
    let maxima = Nx.reduce_segments Max ~segments:3 ids x
    ]}

    Counts are [Add] of ones, a mean is [Add] divided by the counts, and each
    segment's first row is [Min] of the row positions, then a {!take}.

    Raises [Invalid_argument] if [segments] is negative, [x] is 0-d, [ids] is
    not 1-d with one id per row of [x], or [c] is [Add] and [x] is boolean. *)

val reduce_ranges :
  combine ->
  lo:'d int64_t ->
  hi:'d int64_t ->
  ('v, 's, 'd) t ->
  ('v, 's, 'd) t
(** [reduce_ranges c ~lo ~hi x] is one row per range, row [i] the rows of [x]
    from [lo.{i}] included to [hi.{i}] excluded folded by [c]. Bounds clip into
    [\[0, n\]] for [n] rows, so a range may reach past either end of [x], and
    one whose clipped [lo] is at or after its clipped [hi] holds the identity.

    A row's bits depend on its range's bounds and rows alone: rows outside it
    never enter it, so a large row outside a range never cancels into its sum.
    A growing window, a running fold, is the ranges from [0] to [i + 1]. Its
    work is [O((n + m) log n)] for [m] ranges, and its memory [O(n + m)].

    {[
    let x = Nx.create Nx.float64 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    let lo = Nx.create Nx.int64 [| 4 |] [| -2L; 0L; 3L; 4L |]
    let hi = Nx.create Nx.int64 [| 4 |] [| 1L; 3L; 3L; 9L |]
    (* [|1.; 6.; 0.; 5.|] and [|1.; 3.; neg_infinity; 5.|] *)
    let sums = Nx.reduce_ranges Add ~lo ~hi x
    let maxima = Nx.reduce_ranges Max ~lo ~hi x
    ]}

    Raises [Invalid_argument] if [x] is 0-d, [lo] and [hi] are not 1-d of one
    length, or [c] is [Add] and [x] is boolean. *)

val map_segments :
  segments:int ->
  'd int64_t ->
  ('d int64_t -> ('v, 's, 'd) t -> ('w, 'r, 'd) t) ->
  ('v, 's, 'd) t ->
  ('w, 'r, 'd) t
(** [map_segments ~segments ids f x] is [f] applied to each position alone,
    the per-position map: at each position [p] of [ids], the row [f] gives
    [p]'s row of [x] with owner [ids.{p}] where [ids.{p}] is in
    [\[0, segments)], and zeros otherwise, whatever [x] holds at [p]. [f] is
    called once on the rows grouped by id, so what [f] reads by owner is read
    once per block, where a per-position map reads it once per position.

    {b Shapes.} Positions are the indices of [ids], of shape [[| s… |]]. The
    first [ndim ids] axes of [x] broadcast to [[| s… |]], and [p]'s row is [x]
    at [p] after that broadcast, of shape [[| r… |]]. The result has shape
    [[| s…; r'… |]]. Ids may repeat. Results agree with the per-position map
    up to how [f] rounds rows it computes together.

    {b What [f] receives.} [f owners rows] is called exactly once. [rows] is
    [[| g; c; r… |]] at [x]'s dtype, [g] blocks of [c] rows, and [owners] is
    [[| g |]], every owner in range. Every row of block [b] is the row of an
    in-range position whose id is [owners.{b}]: each such position fills one
    row, and the block's other rows repeat rows it already holds. A block no
    position fills takes the owner and row of an in-range position of its
    group. A position whose id is out of range never reaches [f]. In a group
    with no in-range position, every row is zeros. The order of blocks, and
    of rows in a block, is unspecified. [f] returns [[| g; c; r'… |]], with
    [r'] independent of [g] and [c], and computes each row of its result from
    that row and its block's owner alone.

    {b Selection.} Results are selected, never multiplied by a mask: what [f]
    gives a repeated row never reaches the result, and a position out of range
    reaches neither [f] nor the result, whatever its row holds.

    {b Groups and blocks.} Positions are grouped per device. [g] and [c]
    depend on shapes, [segments] and placement, never on the ids' values, so
    the shapes [f] sees never depend on the routing. [c] is [1] when a group
    holds few positions per segment, as one token over many experts: then each
    position is its own block and nothing is sorted. With [segments] zero or
    no positions, [g] is [0] and the result is zeros. [ids] and [x] may be
    split over devices along their first axis only, identically or one of them
    whole on each device; [rows], [owners] and [f]'s result are split along
    their first axis as the positions are.

    {[
    let ids = Nx.create Nx.int64 [| 4 |] [| 1L; -1L; 0L; 1L |]
    let x = Nx.create Nx.float64 [| 4; 1 |] [| 1.; 2.; 3.; 4. |]
    let scale = Nx.create Nx.float64 [| 2; 1; 1 |] [| 10.; 100. |]
    (* [|100.; 0.; 30.; 400.|] *)
    let y =
      Nx.map_segments ~segments:2 ids
        (fun owners rows -> Nx.mul rows (Nx.take ~axis:0 owners scale))
        x
    ]}

    Raises [Invalid_argument] if [segments] is negative, [x] has fewer axes
    than [ids] or its leading axes do not broadcast to [ids]', [ids] or [x] is
    split along another axis or in another number of windows, or [f]'s result
    does not have shape [[| g; c; … |]]. *)

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
  (** [split ~axis s] cuts [axis] into one window per device of [s], in
      order, as evenly as its extent allows: of [n] elements over [k] devices,
      the first [n mod k] windows hold [n / k + 1] and the others [n / k]. A
      window is empty only where [n < k].

      Raises [Invalid_argument] if [axis] is negative or not below
      {!Nx_array.Layout.max_rank}. *)

  val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
  (** [mesh m cuts] cuts each axis [a] of [(a, names)] over the mesh axes
      [names], major first; a mesh axis no cut names holds copies. An axis cut
      over several mesh axes is cut as {!split} cuts it over as many devices
      as their extents multiply to, numbered major first: over extents [e1]
      and [e2], the device at [i1] along the first and [i2] along the second
      holds window [i1 × e2 + i2].
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
    over [x]'s own memory where [x] already lies there, or where the host
    addresses both [x]'s memory and that device's ({!Rig.reaches});
    otherwise it holds a copy of its window, allocating at most the window's
    bytes. A value of every set is computed at [p]. Across sets it changes the
    brand; within one, the arrangement.

    Raises [Invalid_argument] if [p] cuts an axis [x] does not have, and
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
  (** [normal dt shape] is standard normal draws, by the Box-Muller transform:
      draws [2q] and [2q + 1] in C order are the cosine's and the sine's of one
      transform, of the {!uniform} draws [2q] and [2q + 1] at the compute
      dtype.

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

      It is not exact: a draw takes the first of ten rounds that accepts, each
      accepting more than 95% of proposals, and the one element in about
      [10{^13}] that no round accepts is the distribution's mean from a
      concentration of 1 up, and [(a + 1) U{^1/a}] below.

      Raises [Invalid_argument] if an element of [a] is outside [(0, inf)]. *)

  val beta :
    ?key:'d key -> (float, 's, 'd) t -> (float, 's, 'd) t -> (float, 's, 'd) t
  (** [beta a b] is beta draws on [[0, 1]] of concentrations [a] and [b],
      elementwise over their broadcast shape: [G(a) / (G(a) + G(b))] for two
      independent {!gamma} draws, formed from their logarithms so that two draws
      that underflow keep their ratio. Draw [j] in C order is formed from the
      draws [2j] and [2j + 1] that {!gamma} gives from its key over [a] and [b]
      interleaved. It inherits {!gamma}'s rounds.

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

(** {1:printing Printing} *)

val pp : Format.formatter -> ('v, 's, 'd) t -> unit
(** [pp ppf x] prints [x]'s dtype and shape, [on] its set off the host, then
    its elements as nested rows, as [float32 [2; 3] [[1, 2, 3], [4, 5, 6]]].
    Elements print as {!Dtype.pp_value} does: a float in the shortest form that
    reads back to it in its format, an unsigned integer unsigned. Past 1000
    elements an axis longer than 6 shows its first and last 3 around [...]. A
    value with elements is read as {!to_array} reads it.

    A value of every set prints [of every set] and the operation that makes it;
    a traced one [traced by] its interpretation; a dead one the function it was
    donated to; one on a lost device the loss. None of these raises. *)

val to_string : ('v, 's, 'd) t -> string
(** [to_string x] is [x] printed by {!pp}. *)

val print : ('v, 's, 'd) t -> unit
(** [print x] prints [x] by {!pp} on standard output, then a newline. *)

val pp_shape : Format.formatter -> int array -> unit
(** [pp_shape ppf s] prints [s] as nx's messages do: [[2; 3]]. *)

val pp_dtype : Format.formatter -> ('v, 's) dtype -> unit
(** [pp_dtype] is {!Dtype.pp}: [float32]. *)

(** {1:errors Errors}

    Every misuse raises [Invalid_argument] whose message is the function the
    user called, a colon, the reason and, where they matter, the operands as
    [dtype shape on set], as in
    [Nx.add: shapes [2; 3] and [4] do not broadcast]. A lost device
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
      device of [p] in order, each that device's window of one shape: along an
      axis [p] cuts, the sum of its windows' extents, and along any other,
      each array's.

      Raises [Invalid_argument] naming [Nx.Repr.of_shards] unless there is
      one array per device of [p], each on its device, all of one rank that
      has every axis [p] cuts, and each array is its device's window of that
      shape. *)

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

    A value of every set reaches a [Values] interpretation's rule as it is, a
    formula with no placement. An [Extent] interpretation's rule receives the
    operations that compute a formula operand first, in dependency order, and
    the operation reads that rule's results in its place; a formula reaches it
    as it is only where the rule returned it. *)
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
    | Padded : {
        x : ('v, 's, 'd) nx;
        fill : 'v;
        pad : Nx_kernel.pad;
      }
        -> 'd load
        (** [x] padded with [fill] and windowed as [pad] says
            ({!Nx_kernel.load}): the loop's shape is the padded one. *)

  (** What a loop's reduction makes of one output of its program. *)
  type ('d, _) reduction =
    | Monoid :
        Nx_kernel.monoid * int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx) reduction
        (** [Monoid (m, k, dt)] folds output [k] by [m] in that output's dtype
            and rounds once to [dt]. *)
    | Moments :
        int * ('v, 's) dtype
        -> ('d, ('v, 's, 'd) nx * ('v, 's, 'd) nx) reduction
        (** The population mean of output [k], then its variance. *)
    | Arg :
        Nx_kernel.extreme * int * ('v, 's) dtype
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

  type ('d, _) fft =
    | C2c : {
        direction : Nx_kernel.direction;
        axes : int array;
        x : (Complex.t, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 's, 'd) nx) fft
        (** The transform [direction] of [x] along each of [axes]. *)
    | R2c : {
        dtype : (Complex.t, 'c) dtype;
        axes : int array;
        x : (float, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 'c, 'd) nx) fft
        (** The forward transform of a real [x] along each of [axes], bins [0]
            to [n/2] along the last; [dtype] has [x]'s precision. *)
    | C2r : {
        dtype : (float, 'r) dtype;
        n : int;
        axes : int array;
        x : (Complex.t, 's, 'd) nx;
      }
        -> ('d, (float, 'r, 'd) nx) fft
        (** The real inverse transform of [n] points along the last of [axes],
            from [n/2 + 1] bins; [dtype] has [x]'s precision. *)

  (** Each constructor is one routine of {!Nx_kernel.S}, its results in that
      routine's order, each of [a]'s dtype but [Lu]'s positions and [Eig]'s
      results. *)
  type ('d, _) linalg =
    | Cholesky : {
        triangle : Nx_kernel.triangle;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx) linalg
    | Lu :
        ('v, 's, 'd) nx
        -> ( 'd,
             ('v, 's, 'd) nx
             * (int64, Dtype.int64_elt, 'd) nx
             * (int64, Dtype.int64_elt, 'd) nx )
           linalg
        (** [lu], [pivots], [perm]. *)
    | Qr : {
        factors : Nx_kernel.factors;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx * ('v, 's, 'd) nx) linalg  (** [q], [r]. *)
    | Svd : {
        vectors :
          ( Nx_kernel.factors,
            ('v, 's, 'd) nx * ('v, 's, 'd) nx,
            'r )
          Nx_kernel.vectors;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx * 'r) linalg
        (** [s], then [u] and [vh]. *)
    | Eigh : {
        vectors : (unit, ('v, 's, 'd) nx, 'r) Nx_kernel.vectors;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx * 'r) linalg  (** [w], then [v]. *)
    | Eig : {
        vectors : (unit, (Complex.t, 'c, 'd) nx, 'r) Nx_kernel.vectors;
        dtype : (Complex.t, 'c) dtype;
        a : ('v, 's, 'd) nx;
      }
        -> ('d, (Complex.t, 'c, 'd) nx * 'r) linalg
        (** [w], then [v], of [dtype], [a]'s {!Nx_kernel.same_precision}. *)
    | Solve_triangular : {
        triangle : Nx_kernel.triangle;
        adjoint : bool;
        unit_diagonal : bool;
        a : ('v, 's, 'd) nx;
        b : ('v, 's, 'd) nx;
      }
        -> ('d, ('v, 's, 'd) nx) linalg

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
            loads, a creation. Its [Coord] nodes read axes of that shape: a
            program that reads one past its rank is refused. *)
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
            {!Nx_kernel.S.reduce} says. Results drop [axes] and are
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
            each index's, inclusive, of [layout]'s shape, as
            {!Nx_kernel.S.scan} says: a running [Moments] is the running mean
            and variance. The scan of a prefix of the axis is the prefix of
            the scan, bit for bit. [prog] reads coordinates as a map's. *)
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
        combine : Nx_kernel.combine;
        unique : bool;
        axis : int;
        idx : (int64, Dtype.int64_elt, 'd) nx;
        updates : ('v, 's, 'd) nx;
        into : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** [into] with each update combined at its own index with axis [axis]
            replaced by [idx]'s element there, in C order of [updates]
            ({!Nx_kernel.S.scatter}); [idx] and [updates] have one shape,
            [into]'s off [axis]. [unique] promises distinct targets. *)
    | Sort : {
        axis : int;
        descending : bool;
        k : int option;
        x : ('v, 's, 'd) nx;
      }
        -> (('v, 's, 'd) nx * (int64, Dtype.int64_elt, 'd) nx) t
        (** [x]'s elements along [axis], stably, in {!Nx_kernel.S.sort}'s
            order, reversed where [descending], then their positions along
            [axis]; with [Some k], the first [k] of each slice. *)
    | Fold : {
        shape : int array;
        pad : Nx_kernel.pad;
        x : ('v, 's, 'd) nx;
      }
        -> ('v, 's, 'd) nx t
        (** The adjoint of a [Padded] load by [pad] of a value of [shape]
            ({!Nx_kernel.S.fold}). *)
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
        order : Nx_kernel.order;
            (** The order a kernel library adds its sums in: the whole
                contraction's ({!contract_order}). Where no kernel computes
                it, its expansion sums rounded products by a reduction, in
                the reduction's order. *)
        batch : (int * int) array;
        contracting : (int * int) array;
        acc : Dtype.any;
            (** A dtype {!Nx_kernel.accumulates} takes. *)
        out : ('v, 's) dtype;
        a : ('a, 'b, 'd) nx;
        b : ('c, 'e, 'd) nx;
        init : ('v, 's, 'd) nx option;  (** Of the result's shape. *)
      }
        -> ('v, 's, 'd) nx t
        (** The contraction of [a] with [b] by the pairs [batch] and
            [contracting], from [init], computed in [acc] and rounded once to
            [out] ({!Nx_kernel.S.contract}): its result C-contiguous, of the
            shape {!Nx_kernel.Shape.contract} gives. *)
    | Fft : ('d, 'r) fft -> 'r t
        (** A Fourier transform ({!Nx_kernel.S.c2c} and its siblings). *)
    | Linalg : ('d, 'r) linalg -> 'r t
        (** A factorisation or a triangular solve ({!Nx_kernel.S.cholesky} and
            its siblings). *)
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

  val contract_order :
    batch:(int * int) array ->
    contracting:(int * int) array ->
    ('a, 'b, 'd) nx ->
    ('c, 'e, 'd) nx ->
    Nx_kernel.order
  (** [contract_order ~batch ~contracting a b] is the order of summation
      ({!Nx_kernel.section-orders}) of a contraction of [a] with [b]: the one
      the kernels of [a]'s set, or of [b]'s where [a] is of every set, choose
      ({!Nx_kernel.S.contract_order}) for its rows, the product of [a]'s extents
      along the axes no pair names, and its columns, [b]'s likewise, each
      [max_int] past it. Where that set has no kernels, as one of devices of
      several kinds, or both are of every set, it is the one nx.cpu chooses. *)

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
      [f] returns or raises. [name] names it in messages, as ["Rune.grad"].

      An operation [rule] receives is the final consumer of the donated
      handles among its operands ({!Nx.donate}), a movement too, since its
      results hold no handle to pass on: their donors die when it is
      received, and [rule] reads the handles' arrays. *)

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
