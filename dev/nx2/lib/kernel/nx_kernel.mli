(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels: what they compute and the contract they implement.

    A kernel library computes nx's operations on the memory of some devices. Its
    [.mli] states what it computes on and which cases it declines, then includes
    {!S}, so that its build checks it against the contract:
    {[
    (** Kernels for host memory. … *)

    include Nx_kernel.S
    ]}
    A kernel takes its attributes, then its destinations, then its operands.
    Destinations of a fixed number are labelled; where its arguments decide
    their number, they come as an array. {!Prog} names what an elementwise
    kernel computes and {!Shape} the shapes of every kernel's destinations.
    In C, [nx_kinds.h] computes each kind, [nx_prog.h] and [nx_kernel.h] hold
    the codes by which C reads a program and the other constructors, and
    [nx_contract_view.h] a contraction view's struct, with no OCaml header, so
    device sources include them; [nx_contract_view_fill.h] fills a view on
    the host. *)

module Prog = Prog
(** Scalar kinds. *)

type ('v, 's) a := ('v, 's) Nx_array.t
type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

(** {1:loops Loops} *)

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
    windows [windows] of the padded operand, as {!Nx_array.Move.Window} does.
    {!Shape.padded} states which paddings an operand takes. *)

(** The type for an operand of a loop and how the loop reads it. *)
type load =
  | Plain : ('v, 's) a -> load  (** The operand through its layout. *)
  | Padded : { x : ('v, 's) a; fill : 'v; pad : pad } -> load
      (** [x] padded with [fill] as [pad] says: the loop's shape is the padded
          one. *)

type monoid = Sum | Prod | Max | Min | Logsumexp
(** The type for monoids a reduction folds with. *)

type extreme = Max | Min
(** The type for extremes a reduction finds with their positions. *)

(** The type for what a reduction computes from the values of one program
    output along the reduced axes, its terms, numbered from [0] in C order of
    the reduced axes' indices.

    - [Monoid Sum] adds the terms from [+0], as {!Prog.Add} does: a sum of [-0]
      terms is [+0]. [Monoid Prod] multiplies them from [1]. Integers wrap.
    - [Monoid Max] and [Monoid Min] are {!Prog.Maximum} and {!Prog.Minimum} of
      the terms: [-0] below [+0], any and all on booleans.
    - [Monoid Logsumexp] is [m + log Σ exp (x - m)], [m] the terms' maximum:
      [-∞] for no term or terms all [-∞], [+∞] where a term is [+∞] and none is
      NaN.
    - [Moments] is the terms' population mean, then their variance; NaN for no
      term. With an infinite term and no NaN term, the mean is the sum of the
      infinities, a NaN where both signs occur, and the variance a NaN.
    - [Arg e] is the extreme [e] of the terms, as [Monoid Max] or [Monoid Min]
      finds it, then its position: the first term with its bits, as an
      [int64].

    A NaN result, where a term it reduces is a NaN, is the first such term, its
    bits unchanged; with no NaN term it is the quiet NaN of a clear sign bit,
    the NaN {!Prog}'s kinds make from numbers on every target. A float [Sum]
    or [Prod] associates in an order that is a function of the shapes alone
    ({!section-orders}), and each kernel library states its order, the
    bounds of [Logsumexp], reduced and scanned, and those of [Moments]
    reduced. A float sum of [n] terms is within [γ(n - 1) Σ|x|] of the exact
    sum, a product within a relative [γ(n - 1)] barring overflow and
    underflow, where [γ(k) = k u / (1 - k u)] and [u] is the dtype's unit
    roundoff.

    A [Moments] scan gives the same bits in every library that computes
    one. On each slice, a term [t] is the state [(t - c, +0, 1)] of a mean
    [μ], a sum [S] of squared deviations and a count [n], [c] the slice's
    first term, or [0] where that term is not finite. The states combine by
    the odd/even network: neighbours [2j] and [2j + 1] combine into pairs,
    the pairs scan by the same network, result [2j + 1] is the pairs' result
    [j], and result [2j], for [j > 0], is the pairs' result [j - 1] combined
    with state [2j]. A state [a] then a state [b] combine as
    [n = na + nb], [δ = μb - μa], [w = nb / n], [μ = μa + δ w] and
    [S = (Sa + Sb) + (δ δ) (na w)], each operation in the output's dtype as
    {!Prog} computes it. Where [μa] or [μb] is not finite, [μ] is [μa] where
    a term of [a] is a NaN, else [μb] where one of [b] is, else [μa + μb].
    Result [i] is the mean [c + μ] and the variance [S / (i + 1)], or
    [μ - μ] where [μ] is not finite. So the first NaN term makes both
    results its bits from there on, and with no NaN an infinity makes the
    mean the sum of the infinities so far and the variance a NaN. Result
    [i] passes through at most [D = max 0 (2⌈log₂ (i + 1)⌉ - 1)]
    combinations. Barring overflow, for [i + 1] exact in the dtype, [R] the
    range of the terms up to [i] and [V] their variance, the mean [μ] is
    within [(4 D + 1) u R + u|μ|] of the exact one and the variance within
    [(7 D + 1) u V + (4 D² - 2 D) u R √V], to first order in [u]. *)
type reduction = Monoid of monoid | Moments | Arg of extreme

val accepts : reduction -> ('v, 's) Nx_array.Dtype.t -> bool
(** [accepts r dt] is [true] iff [r] reduces terms of [dt]: [Sum] and [Prod]
    every dtype but booleans; [Max], [Min] and [Arg] every dtype; [Logsumexp]
    and [Moments] floats. *)

(** {2:orders Orders of summation}

    Two orders of a float sum, which a kernel library states it follows, so
    that libraries that state the same order give the same bits. A
    contraction's terms are each output's products, a reduction's its terms,
    numbered from [0] by the contracting pairs' or the reduced axes' indices,
    taken in C order. A contraction's product is fused into the addition that
    takes it. A complex contraction sums each part of an output as a float
    sum of products of parts: with [a = ar + i·ai] and [b = br + i·bi], its
    product [p]'s terms are numbered [2p] and [2p + 1], [ar·br] and
    [ai·(-bi)] in the real part, [ar·bi] and [ai·br] in the imaginary
    part.

    {e Chain order} adds the terms in increasing number, from [init] or [+0].

    {e Lane order} puts the terms into blocks of 1024 consecutive numbers and,
    within a block, into 16 lanes by number modulo 16, each lane adding its
    terms in increasing number from [+0] ([1] for a product). Lane [i] takes
    lane [i + 8] for [i] below 8, then lane [i + 4] for [i] below 4, then
    [i + 2] for [i] below 2, then lane 0 takes lane 1. The blocks are combined
    by a binary tree whose left part holds the largest power of two of blocks
    below their count, each part combined the same way; then [init] is added.
    *)

type order =
  | Chain  (** {e Chain order}. *)
  | Lane  (** {e Lane order}. *)
(** The type for the order of summation a contraction is asked for. A library
    that states it follows {!section-orders} adds in it; another states the
    association it computes for each. *)

(** {1:scatters Scatters} *)

type combine = Set | Add | Max | Min
(** The type for how a scatter combines an element with its target's. *)

(** {1:contraction Contraction} *)

val accumulates : ('v, 's) Nx_array.Dtype.t -> bool
(** [accumulates dt] is [true] iff a contraction computes in [dt]: every
    dtype but booleans and floats narrower than [float32]. *)

(** {1:fft Fourier transforms} *)

(** The type for a transform's direction. Along an axis of [n] points,
    [Forward] is [y[k] = Σ_j x[j] e^(-2πi jk/n)] and [Inverse] the same sum
    with [e^(+2πi jk/n)]. Neither divides by [n]. *)
type direction = Forward | Inverse

val same_precision :
  ('v, 's) Nx_array.Dtype.t -> (Complex.t, 'c) Nx_array.Dtype.t -> bool
(** [same_precision x c] is [true] iff [x] and [c] have one precision, the
    dtypes a real transform and {!S.eig} pair: [float32] or [complex64]
    with [complex64], [float64] or [complex128] with [complex128]. *)

(** {1:matrices Matrices} *)

(** The type for a square matrix's triangles, each with its diagonal. *)
type triangle = Lower | Upper

(** The type for how much of a matrix's unitary factors {!Shape.qr} and
    {!Shape.svd} shape, for [m] rows, [n] columns and [k = min m n]: [k]
    columns of the left one and [k] rows of the right one, or the square ones,
    of [m] and [n]. A kernel reads which from its destinations' shapes. *)
type factors = Reduced | Complete

(** The type for the vectors a routine computes beside its values, ['x] their
    results: none, or [Vectors p], [p] how much of them where the routine
    has a choice. *)
type ('p, 'x, _) vectors =
  | No_vectors : ('p, 'x, unit) vectors
  | Vectors : 'p -> ('p, 'x, 'x) vectors

val full_precision : ('v, 's) Nx_array.Dtype.t -> bool
(** [full_precision dt] is [true] iff [dt] is [float32], [float64],
    [complex64] or [complex128], the dtypes the matrix routines,
    {!S.cholesky} to {!S.solve_triangular}, take. *)

(** {1:shapes Shapes} *)

(** Shapes of destinations.

    Each function is the shapes of a kernel's destinations for operands of
    the shapes it takes, or why they do not fit: what the kernel's caller
    allocates, and what a kernel library refuses with [Shape_mismatch] where
    they differ. Every shape it derives, a padded operand's and each
    destination's, is one a layout admits ({!Nx_array.Layout.max_numel}): one
    past it is an [Error]. Axes are checked against the ranks they index. *)
module Shape : sig
  type t = int array
  (** The type for shapes. *)

  type 'a result := ('a, string) Stdlib.result

  (** {1:loops Loops} *)

  val padded : pad -> t -> t result
  (** [padded p x] is the shape a [Padded] load by [p] of an operand of shape
      [x] has. It is an [Error] unless [p]'s [lo], [hi] and [interior] have
      [x]'s rank, [interior] is not negative, each padded extent is not
      negative, and [windows] are on strictly increasing axes of the padded
      shape, with [size], [step] and [dilation] at least [1], each window
      fitting its axis. *)

  val loaded : load array -> t result
  (** [loaded ls] is the one shape the loads [ls] have once loaded. A loop
      with no load has no shape of its own, and is an [Error]. *)

  val reduce : axes:int array -> reduction array -> t -> t result
  (** [reduce ~axes rs y] is the shape of each destination of reductions
      [rs] along [axes] of a loop of shape [y]: [y] without [axes]. It is an
      [Error] unless [axes] are strictly increasing axes of [y], and where a
      [Monoid Max], [Monoid Min] or [Arg] of [rs] has a destination element
      and no term. *)

  val scan : axis:int -> t -> t result
  (** [scan ~axis y] is the shape of a scan's destinations along [axis] of a
      loop of shape [y]: [y]. *)

  (** {1:index Gathers, scatters and sorts} *)

  val gather : axis:int -> t -> t -> t result
  (** [gather ~axis idx x] is the shape of a gather of [x] at positions of
      shape [idx]: [idx], which has [x]'s rank and extents but along [axis]. *)

  val scatter : axis:int -> t -> t -> t -> t result
  (** [scatter ~axis into idx updates] is the shape of a scatter into [into]:
      [into]. [idx] and [updates] have one shape, [into]'s but along
      [axis]. *)

  val sort : axis:int -> k:int option -> t -> t result
  (** [sort ~axis ~k x] is the shape of a sort's [values] and [positions]: [x]
      with [k] elements along [axis] with [Some k], [x] with [None]. It is an
      [Error] for a [k] past that axis's extent or negative. *)

  (** {1:assembly Assemblies} *)

  val assemble : t -> Nx_array.Move.range array array -> t array result
  (** [assemble y regions] is the shape of each piece of an assembly of
      shape [y]: its region's, the elements [regions.(i)] keep of each axis
      of [y], as {!Nx_array.Move.Slice} keeps them. A fold's operand has the
      shape {!padded} gives for its destination's. *)

  (** {1:contraction Contraction} *)

  val contract :
    batch:(int * int) array -> contracting:(int * int) array -> t -> t -> t result
  (** [contract ~batch ~contracting a b] is the shape of a contraction of [a]
      with [b]: the batch axes in pair order, then [a]'s free axes, then
      [b]'s, in axis order. It is an [Error] unless each pair names an axis
      of [a] and an axis of [b] of one extent, and no axis is in two pairs. A
      contraction's [init] has its destination's shape. *)

  (** {1:fft Fourier transforms} *)

  val c2c : axes:int array -> t -> t result
  (** [c2c ~axes x] is [x], where [axes] are strictly increasing axes of [x],
      not empty. *)

  val r2c : axes:int array -> t -> t result
  (** [r2c ~axes x] is [x] but [n/2 + 1] bins along the last of [axes], [n]
      [x]'s extent there, with [axes] as {!c2c} takes them. *)

  val c2r : axes:int array -> n:int -> t -> t result
  (** [c2r ~axes ~n x] is [x] but [n] points along the last of [axes], with
      [axes] as {!c2c} takes them, where [x] has [n/2 + 1] bins there and [n]
      is not negative. *)

  (** {1:linalg Matrices}

      Each takes [a] of rank at least two, its last two axes [m] rows and [n]
      columns, [k = min m n]; its other axes, the batch, lead each shape it
      gives. *)

  val cholesky : t -> t result
  (** [cholesky a] is [a], square. *)

  val lu : t -> (t * t * t) result
  (** [lu a] is the shapes of [lu], [pivots] and [perm]: [a], [[…; k]] and
      [[…; m]]. *)

  val qr : factors -> t -> (t * t) result
  (** [qr f a] is the shapes of [q] and [r]: [[…; m; k]] and [[…; k; n]]
      under [Reduced], [[…; m; m]] and [[…; m; n]] under [Complete]. *)

  val svd : (factors, t * t, 'r) vectors -> t -> (t * 'r) result
  (** [svd vs a] is the shape of [s], [[…; k]], then with [Vectors f] those
      of [u] and [vh]: [[…; m; k]] and [[…; k; n]] under [Reduced],
      [[…; m; m]] and [[…; n; n]] under [Complete]. *)

  val eigh : (unit, t, 'r) vectors -> t -> (t * 'r) result
  (** [eigh vs a] is the shape of [w] of [a], square, [[…; n]], then with
      [Vectors ()] that of [v]: [a]. *)

  val eig : (unit, t, 'r) vectors -> t -> (t * 'r) result
  (** [eig vs a] is the shapes {!eigh} gives. *)

  val solve_triangular : t -> t -> t result
  (** [solve_triangular a b] is [b], [[…; n; r]], for [a] square of [n] and
      [b] of [a]'s batch. *)
end

(** {1:contract The contract} *)

(** The type for kernel libraries.

    Every kernel writes its results into its destinations: C-contiguous,
    reaching no element twice, of the shapes {!Shape} gives and the dtypes its
    doc states, on the operands' device. Operands may be strided, broadcast or
    offset. The caller has checked their shapes, dtypes, axes and attributes
    against the kernel's rule.

    A destination shares no byte with an operand, except one it is identical
    to ({!Nx_array.door}) that the kernel reads only at the result's own index:
    an operand of [apply1] to [apply3], a [Plain] load of [map], [contract]'s
    [init], [scatter]'s [into], or [assemble]'s first piece where its region
    is the whole result. A kernel reads such an operand at an index before it
    writes there, so the result is the one a fresh destination receives. The
    door refuses an operand that shares a byte with a destination without
    being identical to it; an identical operand the kernel reads at other
    indices, as a gather's source or a contraction's [a] and [b], is the
    caller's error.

    A kernel claims its destinations and operands for the extent of its call
    through a door: [nx_read] of [nx_array.h] for host kernels, {!Nx_array.door}
    for kernels that submit device work. [nx_read] waits, under its claims, for
    earlier device work on the operands; {!Nx_array.door} holds its claims until
    the work is submitted. A kernel answers [Done] once its work is done on the
    host or queued on the device's timeline; the door's refusal, before any
    write; [Declined] if it does not compute the case, before any write or
    queued work. Value-dependent failures are NaN. A kernel's result is a
    function of its operands' values alone: neither layouts nor threads change a
    bit. Nor do windows: nx computes an operation over a set one window of its
    results at a time, each device over the parts of the operands its window
    reads, and every element keeps the bits the whole operation gives it. Its
    association, its algorithm and whether the kernel declines depend on the
    extents the element reads whole (a reduction's, a scan's, a sort's or a
    transform's axes, the contracted extents, a routine's matrices, a gather's
    or a scatter's axis, the axes a padded load or a fold pads or windows), the
    dtypes, the attributes and the device's architecture, never on the result's
    other extents. Neither rule covers the sign and payload of a NaN result
    where the operation leaves them unspecified, as a contraction and a
    transform do: IEEE 754 specifies neither for arithmetic, and where two NaNs
    meet, the one an instruction passes on depends on its operands' order,
    which a kernel's paths for different layouts and windows may set
    differently. A kernel raises [Out_of_memory] if host memory runs out and
    {!Rig.Out_of_memory} if its device's does.

    A kernel answers [Wrong_dtype] for dtypes its doc does not take. A decline
    of [apply0] to [apply3] at a base dtype (float32, float64, the 8- to 64-bit
    integers and bool), of [reduce] or [scan] of one [Sum], [Prod], [Max] or
    [Min] of a program's one operand into its own dtype, read plain, at a base
    dtype, of [gather] at a dtype of eight bits or more, or of [scatter] with
    [unique] at such a dtype, is an error its caller raises; elsewhere its
    caller computes the case from other operations. *)
module type S = sig
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  (** [name] names the kernels in messages, as ["nx.cpu"]. *)

  val computes_on : Rig.t -> bool
  (** [computes_on d] is [true] iff these kernels compute on [d]'s memory. *)

  (** {1:elementwise Elementwise} *)

  val apply0 : Prog.op0 -> dst:('v, 's) a -> answer
  (** [apply0 k ~dst] stores [k] at each index of [dst]. *)

  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
  (** [apply1 k ~dst x] stores [k] of each element of [x] into [dst]. *)

  val apply2 : Prog.op2 -> dst:('v, 's) a -> ('a, 'b) a -> ('a, 'b) a -> answer
  (** [apply2 k ~dst x y] stores [k] of each element of [x] and [y] into
      [dst]. *)

  val apply3 :
    Prog.op3 -> dst:('v, 's) a -> ('c, 'e) a -> ('v, 's) a -> ('v, 's) a -> answer
  (** [apply3 k ~dst c x y] stores [k] of each element of [c], [x] and [y] into
      [dst], of [x]'s and [y]'s dtype, as [Where] and [Fma] give. *)

  (** {1:loops Loops}

      A loop runs a program at every index of one shape, the shape its loads
      have once loaded ({!Shape.loaded}), reading load [i] as the program's
      operand [i]. Its loads are one per operand of the program, each of that
      operand's dtype. Its [Coord] nodes read axes of that shape: a loop
      that computes a program with one past its rank answers
      [Shape_mismatch] instead. *)

  val map : Prog.t -> dsts:any array -> load array -> answer
  (** [map p ~dsts loads] stores [p]'s output [k] at each index into
      [dsts.(k)], of that output's dtype. *)

  val reduce :
    Prog.t ->
    axes:int array ->
    (reduction * int) array ->
    dsts:any array ->
    load array ->
    answer
  (** [reduce p ~axes rs ~dsts loads] reduces, for each [(r, k)] of [rs],
      output [k] of [p] along [axes] by [r], in that output's dtype, rounded
      once to its destination's. Destinations follow [rs]: one for a [Monoid],
      the mean then the variance for [Moments], the extreme then its [int64]
      position for [Arg]. Each has the shape {!Shape.reduce} gives. It answers
      [Wrong_dtype] where {!accepts} is [false] of a reduction and its
      output's dtype. *)

  val scan :
    Prog.t -> axis:int -> reduction * int -> dsts:any array -> load array -> answer
  (** [scan p ~axis (r, k) ~dsts loads] stores at each index the reduction [r]
      of output [k] of [p] over the indices along [axis] up to that index's,
      inclusive, in that output's dtype, rounded once to its destination's: a
      running sum for [Monoid Sum], the running population mean and variance
      for [Moments]. The result at an index is a function of the terms up to
      it alone: the scan of a prefix of the axis is the prefix of the scan,
      bit for bit. A scan's last result need not equal the reduction of the
      same terms. Destinations are as {!reduce}'s, each of the loaded shape
      ({!Shape.scan}). It answers [Wrong_dtype] where {!accepts} is
      [false] of [r] and output [k]'s dtype. *)

  (** {1:index Gathers and scatters} *)

  val gather : axis:int -> dst:('v, 's) a -> index -> ('v, 's) a -> answer
  (** [gather ~axis ~dst idx x] stores into [dst] at each index [j] [x]'s
      element at [j] with axis [axis] replaced by [idx]'s element at [j]. A
      position outside \[[0], [d]), [d] [x]'s extent along [axis], reads the
      element of zero bits: [+0], [false], [0 + 0i]. *)

  val scatter :
    combine ->
    unique:bool ->
    axis:int ->
    dst:('v, 's) a ->
    into:('v, 's) a ->
    index ->
    ('v, 's) a ->
    answer
  (** [scatter c ~unique ~axis ~dst ~into idx updates] stores into [dst]
      [into] with [updates] combined at the positions [idx] holds. Update [j]
      targets [into]'s element at [j] with axis [axis] replaced by [idx]'s
      element at [j]; a position outside \[[0], [d]), [d] [into]'s extent
      along [axis], drops it. Where no update lands, the result is [into]'s
      element, its bits unchanged. Where the updates [u1], …, [un], in C order
      of their indices, land on [x], it is:
      - for [Set], [un];
      - for [Add], [+0 + x + u1 + … + un] by {!Prog.Add}, a narrow float's sum
        computed in float32 and rounded once. Each kernel library states how
        it associates the sum, a function of [n] alone;
      - for [Max] and [Min], {!Prog.Maximum} and {!Prog.Minimum} of [x], [u1],
        …, [un]; a NaN result is the first NaN among them, its bits unchanged.

      [unique] promises that no two updates share a target; where two do, the
      result there is unspecified. It answers [Wrong_dtype] for an [Add] of a
      dtype {!Prog.Add} does not take. *)

  (** {1:sorts Sorts} *)

  val sort :
    axis:int ->
    descending:bool ->
    values:('v, 's) a ->
    positions:index ->
    ('v, 's) a ->
    answer
  (** [sort ~axis ~descending ~values ~positions x] orders each slice of [x]
      along [axis], stably, into [values], bits unchanged, and their [int64]
      positions along [axis] into [positions]. Where [values] is shorter than
      [x] along [axis], it keeps the first elements. The order is the one
      {!Prog}'s domains state, [-0] below [+0], with every NaN, and every
      complex number with a NaN part, above [+∞] and equal to each other:
      [-∞ < … < -0 < +0 < … < +∞ < NaN]. [descending] reverses the order and
      keeps equal elements in increasing position. *)

  (** {1:assembly Assemblies and folds} *)

  val assemble :
    fill:'v ->
    dst:('v, 's) a ->
    (Nx_array.Move.range array * ('v, 's) a) array ->
    answer
  (** [assemble ~fill ~dst pieces] stores into [dst], at each index, the
      element at that index of the last piece whose region holds it, or [fill]
      where none does. Each piece has the shape of its region
      ({!Shape.assemble}). Bits are copied, NaN payloads included. *)

  val fold : pad -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [fold p ~dst x] stores the adjoint of a [Padded] load by [p] of [dst]'s
      shape: [x] has the shape that load gives, and [dst]'s element at an
      index is [+0] plus, by {!Prog.Add} from left to right, each element of
      [x] the load reads from that index, in C order of their taps: their
      indices along the axes the load's windows append. It answers
      [Wrong_dtype] for a dtype {!Prog.Add} does not take. *)

  (** {1:contraction Contraction} *)

  val contract_order : rows:int -> columns:int -> order
  (** [contract_order ~rows ~columns] is the order these kernels choose for a
      contraction of [rows] rows, the product of [a]'s extents along the axes
      no pair names, by [columns] columns, [b]'s likewise. nx asks it once of
      the whole contraction, whatever part of it each call computes, so that
      every part sums in the whole's order. *)

  val contract :
    order:order ->
    batch:(int * int) array ->
    contracting:(int * int) array ->
    acc:Nx_array.Dtype.any ->
    dst:('v, 's) a ->
    init:('v, 's) a option ->
    ('a, 'b) a ->
    ('c, 'e) a ->
    answer
  (** [contract ~order ~batch ~contracting ~acc ~dst ~init a b] stores into
      [dst] the contraction of [a] with [b], of the shape {!Shape.contract}
      gives:
      {[
      dst[B, I, J] = init[B, I, J] + Σ_K a[B, I, K] · b[B, K, J]
      ]}
      computed in [acc], a dtype {!accumulates} takes, and rounded once to
      [dst]'s dtype. Without [init] the sum starts from [+0]. Integers wrap in
      [acc]. A float or complex [acc] errs by at most twice its unit
      roundoff, its parts' for a complex one, per addition, barring
      underflow.

      A float sum's association, which fixes its bits, is a function of
      [order], the dtypes and the contracted extent alone: the batch, row and
      column extents, the layouts and the device's size never enter it, so a
      window of [dst] computed alone has the whole's bits. The kernels of one
      order never compute a call given the other. A library that states it
      follows {!section-orders} adds each output's products in [order].
      Another states the association it computes for each order. A NaN
      result's sign and payload are unspecified. *)

  (** {1:fft Fourier transforms}

      Each transform runs along each of [axes], strictly increasing and not
      empty. A transform of no point is the empty sum. Each kernel library
      states its error bound. A NaN result's sign and payload are
      unspecified. *)

  val c2c :
    direction -> axes:int array -> dst:(Complex.t, 's) a -> (Complex.t, 's) a -> answer
  (** [c2c d ~axes ~dst x] stores the transform [d] of [x] into [dst]. *)

  val r2c : axes:int array -> dst:(Complex.t, 'c) a -> (float, 'r) a -> answer
  (** [r2c ~axes ~dst x] stores the [Forward] transform of [x], bins [0] to
      [n/2] along the last of [axes], [n] [x]'s extent there: one bin of [+0]
      for [n = 0]. [x] and [dst] are of the {!same_precision}. *)

  val c2r : axes:int array -> dst:(float, 'r) a -> (Complex.t, 'c) a -> answer
  (** [c2r ~axes ~dst x] reads [n/2 + 1] bins [X] along the last of [axes], [n]
      [dst]'s extent there. It is the [Inverse] transform along each of [axes]
      but the last, then along the last the real parts of the [Inverse]
      transform of [X'], [X'[k]] [X[k]] for [k <= n/2] and [conj X[n - k]]
      above. The imaginary parts of bin [0] and, for an even [n], of bin [n/2]
      never contribute. It is this linear map for every operand, Hermitian or
      not. [dst] and [x] are of the {!same_precision}. *)

  (** {1:linalg Matrices}

      A routine computes on each matrix of [a], its last two axes, of [m] rows
      and [n] columns, [k = min m n]. Its other axes, the batch, lead every
      operand and destination. Each matrix is computed on its own: one that
      fails is NaN in its own results alone. [aᴴ] is the conjugate transpose.
      Routines take the {!full_precision} dtypes; a real result of a
      complex [a], a singular value or an eigenvalue of [eigh], is complex
      with an imaginary part of [+0]. [eig]'s results are complex of [a]'s
      {!same_precision}, a real [a] or not. Each kernel library states its
      error bounds and the choices below leave open: [eig]'s order of
      eigenvalues; [qr]'s factors, unique only up to a unit factor per column
      of [q] and row of [r] where [r]'s diagonal has no zero, and otherwise
      from the first zero on up to an orthonormal completion; the vectors of
      [svd], [eigh] and [eig], unique only up to a unit factor where their
      value is simple and up to a unitary basis of its space where it repeats;
      and the columns complete factors add, any orthonormal completion.

      [qr] and [svd] compute reduced or complete factors as their
      destinations' shapes state, those {!Shape.qr} or {!Shape.svd} gives
      for one {!factors}. Where a destination's two shapes coincide, as
      [q]'s, [r]'s and [u]'s do for [m <= n] and [vh]'s for [m >= n], they
      name one result: [qr]'s factors are then the same either way, and
      [svd] reads the choice from its other factor. *)

  val cholesky : triangle -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [cholesky t ~dst a] stores [L] with [a = L Lᴴ] under [Lower], or
      [U = Lᴴ] under [Upper], [+0] in the other triangle and its diagonal real
      and positive. It reads [a]'s lower triangle and the real parts of its
      diagonal alone. A matrix whose factorisation meets a pivot that is not
      positive, or NaN, is NaN. *)

  val lu : lu:('v, 's) a -> pivots:index -> perm:index -> ('v, 's) a -> answer
  (** [lu ~lu ~pivots ~perm a] stores into [lu] the unit lower triangular [L]
      below its diagonal, its ones unstored, and the upper triangular [U] on
      and above it. At step [j], rows [j] and [pivots[j]] were interchanged,
      [pivots[j]] the row a scan of column [j] down from row [j] ends at: it
      starts at [j] and moves to a later row only where that row's magnitude,
      [|re| + |im|] on complex, is greater than the current one's. A NaN is
      never greater, and nothing is greater than a NaN. Row [i] of [L U] is
      row [perm[i]] of [a]. A zero pivot stays in [U] and its column of [L] is
      not divided. *)

  val qr : q:('v, 's) a -> r:('v, 's) a -> ('v, 's) a -> answer
  (** [qr ~q ~r a] stores [q] with orthonormal columns and the upper
      triangular [r], [a = q r]. *)

  val svd :
    s:('v, 's) a ->
    ?vectors:(u:('v, 's) a * vh:('v, 's) a) ->
    ('v, 's) a ->
    answer
  (** [svd ~s ~vectors:(~u, ~vh) a] stores [a = u diag s vh], [s] descending
      and not negative, a zero one [+0]. Without [vectors] it stores [s]
      alone. A matrix that holds NaN or an infinity, or on which the
      iteration does not converge, is NaN in every result. *)

  val eigh : w:('v, 's) a -> ?v:('v, 's) a -> ('v, 's) a -> answer
  (** [eigh ~w ~v a] stores the eigenvalues of [a], square and Hermitian as
      its lower triangle and the real parts of its diagonal name it, which
      alone are read, ascending into [w], and orthonormal columns [v] with
      [a v = v diag w]. Without [v] it stores [w] alone. NaN as for [svd],
      over what is read. *)

  val eig :
    w:(Complex.t, 'c) a -> ?v:(Complex.t, 'c) a -> ('v, 's) a -> answer
  (** [eig ~w ~v a] stores the eigenvalues of [a], square, into [w], and
      columns [v] of unit norm with [a v = v diag w]. Without [v] it stores
      [w] alone. NaN as for [svd]. It answers [Wrong_dtype] unless [w] and
      [v] are of [a]'s {!same_precision}. *)

  val solve_triangular :
    triangle ->
    adjoint:bool ->
    unit_diagonal:bool ->
    dst:('v, 's) a ->
    ('v, 's) a ->
    ('v, 's) a ->
    answer
  (** [solve_triangular t ~adjoint ~unit_diagonal ~dst a b] stores [x] with
      [a x = b], or [aᴴ x = b] under [adjoint]. It reads [a]'s triangle [t],
      without its diagonal under [unit_diagonal], whose ones it takes instead.
      A zero on a diagonal it reads makes [x] NaN. *)
end

(** {1:views Contraction views} *)

(** Contractions with their axes grouped.

    A view groups a contraction's axes into one of each kind, where every
    operand lays out each group's axes as one run of strides, as a matrix
    product reads them. A C kernel groups them as {!v} does into
    [nx_contract_view.h]'s [nx_contract_view], on its own stack, with
    [nx_contract_view_fill.h]'s [nx_contract_view_fill]. *)
module Contract_view : sig
  type t
  (** The type for views. *)

  (** The type for a contraction's operands and its result. *)
  type operand = A | B | Init | Dst

  (** The type for a view's axes. [A] has [Batch], [Row] and [Contracted]; [B]
      has [Batch], [Column] and [Contracted]; [Init] and [Dst] have [Batch],
      [Row] and [Column]. *)
  type axis = Batch | Row | Column | Contracted

  val v :
    batch:(int * int) array ->
    contracting:(int * int) array ->
    dst:('v, 's) a ->
    init:('v, 's) a option ->
    ('a, 'b) a ->
    ('c, 'e) a ->
    (t, Nx_array.answer) result
  (** [v ~batch ~contracting ~dst ~init a b] is the view of the axes of [a],
      [b], [init] and [dst], as {!S.contract} takes them. It is [Error
      Shape_mismatch] if the operands do not fit the pairs (ranks the pairs
      do not fit, an axis in two pairs, or extents that differ within a
      group), and otherwise [Error Declined] if a group does not merge into
      one axis. *)

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
