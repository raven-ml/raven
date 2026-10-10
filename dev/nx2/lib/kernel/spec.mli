(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Descriptors.

    A descriptor holds the attributes of one operation of a family, checked when
    it is made: what a kernel needs beyond its operands' arrays. It is the C
    struct of [nx_spec.h] for its family, held in a string, so that C kernels
    read it in place and equal descriptors are equal strings. *)

type 'f t
(** The type for descriptors of the family ['f]. *)

val shapes : 'f t -> int array array -> (int array array, string) result
(** [shapes s ins] is the shapes of [s]'s results for operands of the shapes
    [ins], in the order the family's kernel takes them, or why they do not fit
    [s]. A map's results have the one shape its operands have once loaded, a
    reduction's that shape without its axes, a scan's that shape. A loop with
    no operand has no shape of its own, and is an [Error]; so is a [Max],
    [Min] or [Arg] reduction with a result and no term. A gather's result has
    its positions' shape, a scatter's its [into]'s, a sort's two results its
    operand's with [k] elements along its axis, an [Error] for a [k] past the
    axis's extent; an assembly's and a fold's result has their [shape]. *)

(** {1:loads Loads} *)

type pad = {
  lo : int array;
  hi : int array;
  interior : int array;
  windows : Nx_array.Move.window array;
}
(** The type for paddings of an operand of rank [r], each array of [lo], [hi]
    and [interior] of length [r]. Along axis [i] it puts [lo.(i)] elements
    before the operand's, [hi.(i)] after and [interior.(i)] between
    neighbours; a negative [lo] or [hi] crops instead. Then it takes the
    windows [windows] of the padded operand, as {!Nx_array.Move.Window}
    does. *)

type load =
  | Plain  (** The operand through its layout. *)
  | Padded of { fill : string; pad : pad }
      (** The operand padded with the element of bits [fill] in its dtype
          ({!Prog.bits}), as [pad] says. *)
(** The type for how a loop reads an operand. *)

(** {1:monoids Monoids} *)

type monoid = Sum | Prod | Max | Min | Logsumexp
(** The type for monoids a reduction folds with. *)

type extreme = Max | Min
(** The type for extremes a reduction finds with their positions. *)

type combine = Set | Add | Max | Min
(** The type for how a scatter combines an element with its target's. *)

(** {1:maps Maps} *)

type map = [ `Map ]
(** The family of maps. *)

val map : Prog.t -> loads:load array -> map t
(** [map p ~loads] runs [p] at every index of one shape, reading operand [i]
    as [loads.(i)] says. Result [k] is [p]'s output [k] at each index, of that
    node's dtype.

    Raises [Invalid_argument] unless [loads] has one load per operand of [p],
    each [fill] is an element of its operand's dtype, each [pad]'s arrays have
    one length, [interior] is not negative, and [windows] are on strictly
    increasing axes below that length, with [size], [step] and [dilation] at
    least [1]. *)

val prog : [< `Map | `Reduce | `Scan ] t -> Prog.t
(** [prog s] is [s]'s program. *)

val loads : [< `Map | `Reduce | `Scan ] t -> load array
(** [loads s] is how [s] reads each operand. *)

(** {1:reductions Reductions and scans} *)

(** The type for what a reduction computes from the values of one program
    output along the reduced axes, its terms, numbered from [0] in C order of
    the reduced axes' indices.

    - [Monoid Sum] adds the terms from [+0], as {!Prog.Add} does: a sum of
      [-0] terms is [+0]. [Monoid Prod] multiplies them from [1]. Integers
      wrap.
    - [Monoid Max] and [Monoid Min] are {!Prog.Maximum} and {!Prog.Minimum}
      of the terms: [-0] below [+0], any and all on booleans.
    - [Monoid Logsumexp] is [m + log Σ exp (x - m)], [m] the terms' maximum:
      [-∞] for no term or terms all [-∞], [+∞] where a term is [+∞] and none
      is NaN.
    - [Moments] is the terms' population mean, then their variance; NaN for
      no term.
    - [Arg e] is the extreme [e] of the terms, as [Monoid Max] or
      [Monoid Min] finds it, then its position: the first term with its bits,
      as an [int64].

    A NaN result, where a term it reduces is a NaN, is the first such term,
    its bits unchanged; with no NaN term it is the instruction set's NaN, as
    an infinity less itself gives. A float [Sum] or [Prod] associates in an
    order that is a function of the shapes alone, never of layouts, threads or
    the device's size, and each kernel library states its order and the
    bounds of [Logsumexp] and [Moments]. A float sum of [n] terms is within
    [γ(n - 1) Σ|x|] of the exact sum, a product within a relative [γ(n - 1)]
    barring overflow and underflow, where [γ(k) = k u / (1 - k u)] and [u] is
    the dtype's unit roundoff. *)
type reduction = Monoid of monoid | Moments | Arg of extreme

val accepts : reduction -> Nx_array.Dtype.any -> bool
(** [accepts r dt] is [true] iff [r] reduces terms of [dt]: [Sum] and [Prod]
    every dtype but booleans; [Max], [Min] and [Arg] every dtype;
    [Logsumexp] and [Moments] floats. *)

type reduce = [ `Reduce ]
(** The family of reductions. *)

val reduce :
  Prog.t ->
  loads:load array ->
  axes:int array ->
  (reduction * int * Nx_array.Dtype.any) array ->
  reduce t
(** [reduce p ~loads ~axes rs] runs [p] at every index of one shape, reading
    operand [i] as [loads.(i)] says, and reduces along [axes]. Each
    [(r, k, dt)] of [rs] reduces output [k] of [p] by [r] in that output's
    dtype and rounds its results once to [dt]: one result for a [Monoid], the
    mean and the variance for [Moments], the extreme and its [int64] position
    for [Arg]. The results come in the order of [rs], each of the loaded
    shape without [axes].

    Raises [Invalid_argument] as {!map} does, and unless [axes] are strictly
    increasing, not negative and below {!Nx_array.Layout.max_rank}, [rs] is
    not empty, each [k] names an output of [p] whose dtype [r] accepts, and
    the loads and results number at most {!Prog.max_operands}. *)

val axes : [< `Reduce | `Scan ] t -> int array
(** [axes s] is the axes [s] reduces: one for a scan. *)

val reductions :
  [< `Reduce | `Scan ] t -> (reduction * int * Nx_array.Dtype.any) array
(** [reductions s] is [s]'s reductions: one for a scan. *)

type scan = [ `Scan ]
(** The family of scans. *)

val scan :
  Prog.t ->
  loads:load array ->
  axis:int ->
  reduction * int * Nx_array.Dtype.any ->
  scan t
(** [scan p ~loads ~axis (r, k, dt)] is, at every index of one shape, the
    reduction [r] of output [k] of [p] over the indices along [axis] up to
    that index's, inclusive, rounded once to [dt]: one result of the loaded
    shape, or for [Arg] the running extreme and its position along [axis].

    Raises [Invalid_argument] as {!reduce} does, and if [r] is [Moments]. *)

(** {1:index Gathers and scatters} *)

type gather = [ `Gather ]
(** The family of gathers. *)

val gather : axis:int -> gather t
(** [gather ~axis] reads an operand [x] at positions held in an [int64]
    operand [idx] of [x]'s rank, which has [x]'s extents along every axis but
    [axis]. The result has [idx]'s shape; its element at an index [j] is
    [x]'s at [j] with axis [axis] replaced by [idx]'s element at [j]. A
    position outside \[[0], [d]), [d] [x]'s extent along [axis], reads the
    element of zero bits: [+0], [false], [0 + 0i].

    Raises [Invalid_argument] unless [0 <= axis] and [axis] is below
    {!Nx_array.Layout.max_rank}. *)

type scatter = [ `Scatter ]
(** The family of scatters. *)

val scatter : combine -> unique:bool -> axis:int -> scatter t
(** [scatter c ~unique ~axis] combines the elements of an operand [updates]
    into an operand [into] at positions held in an [int64] operand [idx] of
    [updates]' shape. The three have one rank and, along every axis but
    [axis], one extent; the result has [into]'s shape. Update [j] targets
    [into]'s element at [j] with axis [axis] replaced by [idx]'s element at
    [j]; a position outside \[[0], [d]), [d] [into]'s extent along [axis],
    drops the update. Where no update lands, the result is [into]'s element,
    its bits unchanged. Where the updates [u1], …, [un], in C order of their
    indices, land on [into]'s element [x], it is:
    - for [Set], [un];
    - for [Add], [+0 + x + u1 + … + un] by {!Prog.Add}, a narrow float's sum
      computed in float32 and rounded once. Each kernel library states how
      it associates the sum, a function of [n] alone;
    - for [Max] and [Min], {!Prog.Maximum} and {!Prog.Minimum} of [x], [u1],
      …, [un]; a NaN result is the first NaN among them, its bits unchanged.

    [Add] takes the dtypes {!Prog.Add} takes, the others every dtype.
    [unique] promises that no two updates share a target; where two do, the
    result there is unspecified.

    Raises [Invalid_argument] as {!gather} does. *)

val axis : [< `Gather | `Scatter | `Sort ] t -> int
(** [axis s] is the axis [s] indexes or sorts along. *)

val combine : scatter t -> combine
(** [combine s] is how [s] combines. *)

val unique : scatter t -> bool
(** [unique s] is [true] iff [s] promises distinct targets. *)

(** {1:sorts Sorts} *)

type sort = [ `Sort ]
(** The family of sorts. *)

val sort : axis:int -> descending:bool -> k:int option -> sort t
(** [sort ~axis ~descending ~k] orders each slice of an operand [x] along
    [axis], stably. Its results are the ordered elements, their bits
    unchanged, and their [int64] positions along [axis]. The order is the one
    {!Prog}'s domains state, [-0] below [+0], with every NaN, and every
    complex number with a NaN part, above [+∞] and equal to each other:
    [-∞ < … < -0 < +0 < … < +∞ < NaN]. [descending] reverses the order and
    keeps the sort stable: equal elements stay in increasing position. With
    [Some k] the results keep the first [k] elements along [axis].

    Raises [Invalid_argument] as {!gather} does, and if [k] is negative. *)

val descending : sort t -> bool
(** [descending s] is [true] iff [s] orders by the reversed order. *)

val k : sort t -> int option
(** [k s] is the number of elements [s] keeps along its axis, if not all. *)

(** {1:assembly Assemblies and folds} *)

type assemble = [ `Assemble ]
(** The family of assemblies. *)

val assemble :
  shape:int array -> fill:string -> Nx_array.Move.range array array -> assemble t
(** [assemble ~shape ~fill regions] is an array of shape [shape] whose element
    at an index is the element at that index of the last piece whose region
    holds it, or the element of bits [fill] in the result's dtype
    ({!Prog.bits}) where none does. Piece [i] covers the region
    [regions.(i)], the elements its ranges keep of each axis, as
    {!Nx_array.Move.Slice} keeps them, and has that slice's shape. Bits are
    copied, NaN payloads included.

    Raises [Invalid_argument] unless [shape] has at most
    {!Nx_array.Layout.max_rank} extents, none negative, [fill] one to sixteen
    bytes, the widest element's, and each region is a slice {!Nx_array.Move.shape} takes for
    [shape]. *)

type fold = [ `Fold ]
(** The family of folds. *)

val fold : shape:int array -> pad -> fold t
(** [fold ~shape p] is the adjoint of a [Padded] load by [p] of an array of
    shape [shape]. Its operand [x] has the shape that load gives; the
    result's element at an index is [+0] plus, by {!Prog.Add} from left to
    right, each element of [x] the load reads from that index, in C order of
    their taps: their indices along the axes the load's windows append.

    Raises [Invalid_argument] unless [shape] has at most
    {!Nx_array.Layout.max_rank} extents, none negative, and [p] is a padding
    of [shape] that {!map} takes and whose padded extents are not
    negative. *)

val shape : [< `Assemble | `Fold ] t -> int array
(** [shape s] is the shape of [s]'s result. *)

val fill : assemble t -> string
(** [fill s] is the bits of the element [s] stores where no piece lies. *)

val regions : assemble t -> Nx_array.Move.range array array
(** [regions s] is each piece's region. *)

val pad : fold t -> pad
(** [pad s] is the padding [s] is the adjoint of. *)

(** {1:contract Contractions} *)

type contract = [ `Contract ]
(** The family of contractions. *)

val contract :
  batch:(int * int) array ->
  contracting:(int * int) array ->
  acc:Nx_array.Dtype.any ->
  out:Nx_array.Dtype.any ->
  init:bool ->
  contract t
(** [contract ~batch ~contracting ~acc ~out ~init] contracts an operand [a] with
    an operand [b]. Each pair of [batch] and [contracting] is an axis of [a] and
    an axis of [b] of one extent; the other axes are free. The result's axes are
    the batch axes in pair order, then [a]'s free axes, then [b]'s, in axis
    order:
    {[
    y[B, I, J] = out (init[B, I, J] + Σ_K a[B, I, K] · b[B, K, J])
    ]}
    computed in [acc] and rounded once to [out]. Without [init] the sum starts
    from [+0]. Integers wrap in [acc]. A float [acc] errs by at most twice its
    unit roundoff per addition, and the order of the sum is a function of the
    shapes, never of layouts, threads or the device's size.

    Raises [Invalid_argument] if an axis is negative or not below
    {!Nx_array.Layout.max_rank}, if an axis of [a] or of [b] is in two pairs, or
    if [acc] is a boolean or a float narrower than [float32]. *)

val batch : contract t -> (int * int) array
(** [batch s] is [s]'s batch pairs. *)

val contracting : contract t -> (int * int) array
(** [contracting s] is [s]'s contracting pairs. *)

val acc : contract t -> Nx_array.Dtype.any
(** [acc s] is the dtype [s] computes in. *)

val out : contract t -> Nx_array.Dtype.any
(** [out s] is [s]'s result's dtype. *)

val init : contract t -> bool
(** [init s] is [true] iff [s] adds an operand [init] to its sum. *)

(** Contractions with their axes grouped.

    A view groups a contraction's axes into one of each kind, where every
    operand lays out each group's axes as one run of strides, as a matrix
    product reads them. *)
module Contract_view : sig
  type 'f spec := 'f t

  type t
  (** The type for views, rewritten by {!fill}. A C kernel reads a view as
      [nx_spec.h]'s [nx_contract_view]: its value is bytes that begin with
      that struct. *)

  (** The type for a contraction's operands and its result. *)
  type operand = A | B | Init | Dst

  (** The type for a view's axes. [A] has [Batch], [Row] and [Contracted]; [B]
      has [Batch], [Column] and [Contracted]; [Init] and [Dst] have [Batch],
      [Row] and [Column]. *)
  type axis = Batch | Row | Column | Contracted

  val make : unit -> t
  (** [make ()] is a view to {!fill}. *)

  val fill :
    t -> contract spec -> dst:Nx_array.any -> Nx_array.any array -> bool
  (** [fill v s ~dst ops] groups the axes of [ops], ordered as
      {!Nx_kernel.S.contract} takes them, and of [dst] into [v]. It is [false]
      if a group does not merge into one axis, and [v] is then unspecified. It
      allocates nothing.

      Raises [Invalid_argument] if [ops] and [dst] do not fit [s]: another
      number of operands, ranks the pairs do not fit, or extents that differ
      within a group. *)

  val extent : t -> axis -> int
  (** [extent v x] is [v]'s extent along [x]: [1] for a group of no axis. *)

  val offset : t -> operand -> int
  (** [offset v o] is [o]'s first element, in elements from its buffer's first
      byte.

      Raises [Invalid_argument] for [Init] in a view without it. *)

  val stride : t -> operand -> axis -> int
  (** [stride v o x] is [o]'s stride along [x], in elements.

      Raises [Invalid_argument] for an axis [o] does not have, and for [Init] in
      a view without it. *)
end
