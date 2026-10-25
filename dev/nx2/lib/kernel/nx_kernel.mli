(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels: what they compute and the contract they implement.

    A {e kernel} computes one of nx's operations on arrays of one device. It
    reads its {e operands}, writes its results into its {e destinations},
    arrays its caller allocates, and takes its other arguments as
    {e attributes}, such as the axes it reduces. A {e kernel library} holds
    kernels that compute on the memory of some devices. Its [.mli] states
    what it computes on, which cases it declines, and the algorithms and
    error bounds this contract leaves to it, then includes {!S}, so that its
    build checks it against the contract:
    {[
    (** Kernels for host memory. … *)

    include Nx_kernel.S
    ]}
    A kernel takes its attributes, then its destinations, then its operands.
    Destinations of a fixed number are labelled; where its arguments decide
    their number, they come as an array. {!Prog} names what a loop
    ({!S.section-loops}) computes at each index and {!Shape} the shapes of
    every kernel's destinations.
    In C, [nx_kinds.h] computes each kind, [nx_kinds_real.h] and
    [nx_kinds_complex.h] holding its float and complex kinds' bodies,
    [nx_prog.h] and [nx_kernel.h] hold the codes by which C reads a program
    and the other constructors, and [nx_contract_view.h] a contraction view's
    struct, with no OCaml header, so that C, CUDA, HIP and Metal sources
    include them; [nx_contract_view_fill.h] fills a view on the host. *)

module Prog = Prog
(** Scalar kinds. *)

type ('v, 's) a := ('v, 's) Nx_array.t
type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

(** {1:loops Loops} *)

type pad = {
  lo : int iarray;
  hi : int iarray;
  interior : int iarray;
  windows : Nx_array.Move.window iarray;
}
(** The type for paddings of an operand of rank [r], each array of [lo], [hi]
    and [interior] of length [r]. Along axis [i] it puts [lo.(i)] elements
    before the operand's, [hi.(i)] after and [interior.(i)] between
    neighbours; a negative [lo] or [hi] crops instead. Then it takes the
    windows of the padded operand that {!Nx_array.Move.Window}[ windows]
    takes. {!Shape.padded} states which paddings an operand takes. *)

(** The type for an operand of a loop and how the loop reads it. *)
type load =
  | Plain : ('v, 's) a -> load  (** The operand through its layout. *)
  | Padded : { x : ('v, 's) a; fill : 'v; pad : pad } -> load
      (** [x] padded with [fill] as [pad] says. The loop's shape is the padded
          one. *)

(** {1:reductions Reductions and scans}

    A reduction computes from the values of one program output along the
    reduced axes, its {e terms}, numbered from [0] in C order of the reduced
    axes' indices.

    A float result is a NaN where a term it reduces is a NaN, or where its
    arithmetic makes one from numbers, as [+∞] plus [-∞] does. Which NaN,
    its sign and its payload are unspecified, as IEEE 754 leaves them, so a
    kernel may reduce by a tree and by the hardware's own instructions. To
    every reduction but [Sum], a complex number with a NaN part is a NaN; a
    complex [Sum] adds each part as a float sum.

    A float [Sum] or [Prod] reduction adds or multiplies each output's terms
    in the association its library states, a float narrower than [float32]
    in [float32], and a complex [Prod] multiplies them by {!Prog.Mul}. A
    float sum of [n] terms, a reduction's or a scan's result, is within
    [γ(n - 1) Σ|x|] of the exact sum, a float product within a relative
    [γ(n - 1)], and a complex product within a normwise relative
    [γ(2n - 2)], each {!Prog.Mul} erring by at most [2u], barring overflow
    and underflow, where [γ(k) = k u / (1 - k u)] and [u] is the dtype's
    unit roundoff, of its parts' dtype for a complex one. *)

type monoid = Sum | Prod | Max | Min
(** The type for monoids a reduction folds with. *)

type extreme = Max | Min
(** The type for extremes a reduction finds with their positions. *)

val carrier : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [carrier dt] is the dtype kernels reduce and scan terms of [dt] in, and a
    scan's carry's: [float32] for a float narrower than it, [int8] and [uint8]
    for [int4] and [uint4], [bool] for [bit], else [dt]. *)

(** The type for a float dtype, whatever its storage. *)
type any_float = Any_float : (float, 's) Nx_array.Dtype.t -> any_float
[@@unboxed]

val float_carrier : (float, 's) Nx_array.Dtype.t -> any_float
(** [float_carrier dt] is {!carrier} of the float [dt]. It is the dtype a
    float function of [dt] computes in before it rounds once to [dt]. *)

(** The type for what a reduction computes from its terms. *)
type reduction =
  | Monoid of monoid
      (** [Monoid Sum] adds the terms from [+0], as {!Prog.Add} does, so a
          sum of [-0] terms is [+0]. [Monoid Prod] multiplies them from the
          first, never by [1], which would change a complex term's signed
          zeros or make NaN parts from its infinite ones; the product of no
          term is [1]. Integers wrap. [Monoid Max] and [Monoid Min] are
          {!Prog.Maximum} and {!Prog.Minimum} of the terms, which put [-0]
          below [+0] and are any and all on booleans. *)
  | Logsumexp
      (** [Logsumexp] is [m + log Σ exp (x - m)], [m] the terms' maximum. It
          is [-∞] for no term or terms all [-∞], [+∞] where a term is [+∞]
          and none is NaN, and a NaN where a term is. Each kernel library
          states its algorithm and its bound. *)
  | Moments
      (** [Moments] is the terms' population mean, then their variance; NaN
          for no term or where a term is a NaN. With an infinite term and no
          NaN term, the mean is the sum of the infinities, a NaN where both
          signs occur, and the variance a NaN. The variance is never
          negative. Each kernel library states its algorithm and their
          bounds. *)
  | Arg of extreme
      (** [Arg e] is the extreme [e] of the terms, as [Monoid Max] or
          [Monoid Min] finds it, then its position as an [int64], that of the
          first term with its bits, or of the first NaN term where it is a
          NaN. *)

val accepts : reduction -> ('v, 's) Nx_array.Dtype.t -> bool
(** [accepts r dt] is [true] iff [r] reduces terms of [dt]: [Sum] and [Prod]
    every dtype but booleans; [Max], [Min] and [Arg] every dtype; [Logsumexp]
    and [Moments] floats. *)

(** {2:scans Scans}

    A float or complex [Sum] or [Prod] scan combines the terms up to each
    result by {!Prog.Add} or {!Prog.Mul} in the carrier ({!carrier}), in the
    association its library states, a product from its first term, so that
    its first result is that term. A [Logsumexp] or [Moments] scan's result
    has the special values of that reduction of the terms up to it
    ({!type-reduction}), and each kernel library states its algorithm and
    its bounds. Each result rounds once into its destination. A scan's
    bounds are on the result in the carrier, before the rounding into the
    destination, which adds its own; they are to first order in the
    carrier's unit roundoff [u], barring overflow.

    The {e odd/even network} scans a slice by combining states, one per
    term. Neighbours [2j] and [2j + 1] combine into
    pairs, the pairs scan by the same network, result [0] is state [0],
    result [2j + 1] is the pairs' result [j], and result [2j], for [j > 0],
    is the pairs' result [j - 1] combined with state [2j]. The earlier
    state comes first in each combination, and result [i] passes through
    at most [D = max 0 (2⌈log₂ (i + 1)⌉ - 1)] of them. A library that
    scans by it states its bounds in [D]. *)

(** {1:scatters Scatters} *)

type combine = Set | Add | Max | Min
(** The type for how a scatter combines an element with its target's. *)

(** {1:contraction Contraction}

    A contraction of [a] with [b] pairs axes of [a] with axes of [b] of one
    extent. The result keeps the {e batch} pairs and sums over the
    {e contracting} pairs. An operand's other axes are its {e free} axes,
    [a]'s the result's {e rows} and [b]'s its {e columns}. An output's
    {e products} are those of [a]'s and [b]'s elements at its batch, row and
    column indices, one per index of the contracting pairs. The
    {e contracted extent} is the contracting pairs' extents multiplied. The
    products are the output's terms, numbered in C order of the contracting
    pairs' indices. A product is fused into the addition
    that takes it. A complex contraction sums each part of an output as a
    float sum of products of parts. With [a = ar + i·ai] and
    [b = br + i·bi], its product [p]'s terms are numbered [2p] and [2p + 1],
    [ar·br] and [ai·(-bi)] in the real part, [ar·bi] and [ai·br] in the
    imaginary part. *)

val accumulates : ('v, 's) Nx_array.Dtype.t -> bool
(** [accumulates dt] is [true] iff a contraction computes in [dt]: every
    dtype but booleans and floats narrower than [float32]. *)

(** The type for a block scale of a contraction operand's array: an array of
    its rank, each extent dividing the operand's, [0] only along an empty
    axis. The operand's element at index [i] is in the {e scale block} [i]
    divided by the ratio of the extents, axis by axis. A scale indexes the
    operand's own index, whatever its layout. A kernel sees no stored array
    behind a view, so a view that cuts an axis within its scale blocks has
    no scale. *)
type scale =
  | Exponent of (int, Nx_array.Dtype.int8_unsigned_elt) Nx_array.t
      (** Code [e] scales its scale block by [2^(e - 127)]; [255] makes it
          NaN. *)
  | Value : ('v, 's) Nx_array.t -> scale
      (** Each scale read in its own dtype into the accumulator. *)

(** The type for a contraction's operand: an array, through any layout
    (strided, broadcast, windowed), never padded, and its scale. Its element
    at an index is the array's times its scale block's scale. *)
type operand = { array : Nx_array.any; scale : scale option }

(** The type for a contraction cut into groups by [offsets]: an [int64]
    array for {!S.contract}, its shape for {!Shape.contract}. [offsets] has the
    batch extents, then [g + 1] entries. At each batch index, with [o] its
    entries, group [e < g] holds the segmented axis's indices [r] with
    [max o.(0) o.(e) <= r] and [r] below each of [o.(e + 1)] to [o.(g)];
    the other indices are in no group. Offsets may take any values,
    decreasing, negative or past the axis. *)
type 'i segments =
  | Rows of { offsets : 'i; a_axis : int; b_axis : int }
      (** [a]'s free axis [a_axis] is segmented; [b]'s free axis [b_axis] has
          extent [g] and leaves the result. Index [r] along [a_axis]
          multiplies [b]'s group [e] of [r], its index [e] along [b_axis].
          An output whose index along [a_axis] is in no group has no
          products. *)
  | Contracting of { offsets : 'i; pair : int }
      (** Contracting pair [pair] is segmented on both sides. The result
          gains a leading axis of [g], slice [e] summing group [e]'s indices
          alone. Its products are numbered from its group's first, and the
          pair's extent is the group's count; a group of no index has no
          products. It is the transpose of [Rows] with respect to [b]. *)

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

val full_precision : ('v, 's) Nx_array.Dtype.t -> bool
(** [full_precision dt] is [true] iff [dt] is [float32], [float64],
    [complex64] or [complex128], the dtypes the matrix routines,
    {!S.cholesky} to {!S.solve_triangular}, take. *)

(** {1:shapes Shapes} *)

(** Shapes of destinations.

    Each function is the shapes of a kernel's destinations for operands of
    the shapes it takes, the destinations its caller allocates, or why the
    operands do not fit. A kernel refuses other shapes ({!S}). Every shape
    it derives, a padded operand's and each destination's, is one a layout
    admits ({!Nx_array.Layout.max_numel}), and one past it is an [Error].
    Axes are checked against the ranks they index. *)
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
      shape, with [size], [step] and [dilation] at least [1]. A window longer
      than its axis gives it no window, as {!Nx_array.Move.Window} counts
      them. *)

  val loaded : load array -> t result
  (** [loaded ls] is the one shape the loads [ls] have once loaded.
      [loaded [||]] is an [Error]. *)

  val reduce : axes:int iarray -> reduction array -> t -> t result
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

  val assemble : t -> Nx_array.Move.range iarray array -> t array result
  (** [assemble y regions] is the shape of each piece of an assembly of
      shape [y]: its region's, the elements [regions.(i)] keep of each axis
      of [y], as {!Nx_array.Move.Slice} keeps them. A fold's operand has the
      shape {!padded} gives for its destination's. *)

  (** {1:contraction Contraction} *)

  val contract :
    batch:(int * int) iarray ->
    contracting:(int * int) iarray ->
    segments:t segments option ->
    t * t option ->
    t * t option ->
    t result
  (** [contract ~batch ~contracting ~segments (a, sa) (b, sb)] is the shape of
      a contraction of [a] with [b], [sa] and [sb] their scales' shapes: the
      batch axes in pair order, then [a]'s free axes, then [b]'s, in axis
      order. [Rows] drops [b]'s [b_axis]; [Contracting] puts the groups'
      axis first. It is an [Error] unless each pair names an axis of [a] and
      an axis of [b] of one extent, no axis is in two pairs, each scale fits
      its operand ({!type-scale}) and the segments' offsets have the shape
      {!type-segments} states, with [a_axis] and [b_axis] free axes and
      [pair] a contracting pair. A contraction's [init] has its destination's
      shape. *)

  (** {1:fft Fourier transforms} *)

  val c2c : axes:int iarray -> t -> t result
  (** [c2c ~axes x] is [x], where [axes] are strictly increasing axes of [x],
      not empty. *)

  val r2c : axes:int iarray -> t -> t result
  (** [r2c ~axes x] is [x] but [n/2 + 1] bins along the last of [axes], [n]
      [x]'s extent there, with [axes] as {!c2c} takes them. *)

  val c2r : axes:int iarray -> n:int -> t -> t result
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

  val svd_values : t -> t result
  (** [svd_values a] is the shape of [s]: [[…; k]]. *)

  val svd : factors -> t -> (t * t * t) result
  (** [svd f a] is the shapes of [s], [u] and [vh]: [[…; k]], then
      [[…; m; k]] and [[…; k; n]] under [Reduced], [[…; m; m]] and
      [[…; n; n]] under [Complete]. *)

  val eigh_values : t -> t result
  (** [eigh_values a] is the shape of [w] of [a], square: [[…; n]], for
      {!S.eigh} and {!S.eig} alike. *)

  val eigh : t -> (t * t) result
  (** [eigh a] is the shapes of [w] and [v] of [a], square: [[…; n]] and
      [a], for {!S.eigh} and {!S.eig} alike. *)

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
    a [Plain] load of [map], [contract]'s [init], [scatter]'s [into], or
    [assemble]'s first piece where its region is the whole result. A kernel
    reads such an operand at an index before it writes there, so the result
    is the one a fresh destination receives. The caller must not pass a
    destination identical to an operand the kernel reads at other indices, as
    it reads a gather's source or a contraction's [a] and [b]: the door
    accepts it, and the result is unspecified unless the library's doc
    states it.

    A kernel claims its destinations and operands for the extent of its call
    through a door, [nx_read] of [nx_array.h] for host kernels and
    {!Nx_array.door} for kernels that submit device work. [nx_read] waits,
    under its claims, for earlier device work on the operands;
    {!Nx_array.door} holds its claims until the work is submitted. A kernel
    answers as {!Nx_array.type-answer} states, and [Declined] before any
    write or queued work. Any case may be declined. On a decline nx raises
    [Invalid_argument] naming the library, the operation and its dtypes,
    with the move that computes it, [Nx.place] onto a set whose kernels do,
    and computes no case in the library's place. Value-dependent failures
    are NaN. A kernel raises [Out_of_memory] if host memory runs out and
    {!Rig.Out_of_memory} if its device's does.

    A kernel is {e deterministic}, in the sense that for the same operand
    values, shapes, layouts and dtypes, the same attributes, its library
    and a device of the same model, it gives the same bits on every run.
    No association of
    a float sum is left to atomics or to the order in which threads finish.
    Within the bounds this contract and its library state, a float or
    complex result may otherwise differ between libraries, device models,
    layouts and extents. nx may compute an operation over several devices,
    each computing a {e window} of the results, a box of their indices,
    from the parts of the operands the window reads, or compute a float
    reduction, scan or contraction from partial results over parts of its
    terms, one per device, combined in a fixed order. A result whose bits
    this contract fixes (an integer's or a boolean's, an extreme, a
    position, a move, a map of kinds {!Prog} defines exactly) keeps the bits
    of the whole operation, and another float or complex one the bounds its
    libraries and nx state. Whether a kernel declines depends on the extents
    an element reads whole (a reduction's, a scan's, a sort's or a
    transform's axes, the contracted extent, a routine's matrices, a
    gather's or a scatter's axis, the axes a padded load or a fold pads or
    windows), the dtypes, the attributes and the device model, never on the
    result's other extents. Which NaN a NaN result is, its sign and its
    payload, only the moves of bits specify. IEEE 754 specifies neither for
    arithmetic, and where two NaNs meet, the one an instruction passes on
    depends on its operands' order, which a kernel's paths for different
    layouts may set differently.

    A kernel answers [Wrong_dtype] for dtypes its doc does not take, and
    [Shape_mismatch] for operands or destinations that do not fit it by
    their number or by the shapes {!Shape} gives. *)
module type S = sig
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  (** [name] names the kernels in messages, as ["nx.cpu"]. *)

  val computes_on : Rig.t -> bool
  (** [computes_on d] is [true] iff these kernels compute on [d]'s memory. *)

  (** {1:loops Loops}

      A loop runs a program at every index of one shape, its destinations'
      shape, except a reduction, whose destinations drop [axes] and which is
      given its shape. It reads load [i] as the program's operand [i]. Its
      loads are one per operand of the program, each of that operand's
      dtype, and each has the loop's shape once loaded ({!Shape.loaded}); a
      loop of no load runs over its shape alone. Its [Coord] nodes read axes
      of that shape. A loop answers [Shape_mismatch] for a program with a
      [Coord] past that shape's rank. *)

  val map : Prog.t -> dsts:any array -> load array -> answer
  (** [map p ~dsts loads] stores [p]'s output [k] at each index into
      [dsts.(k)], of that output's dtype. It is the one elementwise kernel.
      One kind over operands comes as the program of one node that
      {!Prog.of_node} makes, which {!Prog.single} and {!Prog.is_single}
      recognise. *)

  val reduce :
    Prog.t ->
    shape:int array ->
    axes:int iarray ->
    (reduction * int) array ->
    dsts:any array ->
    load array ->
    answer
  (** [reduce p ~shape ~axes rs ~dsts loads] reduces, for each [(r, k)] of
      [rs], output [k] of [p] over the loop of shape [shape] along [axes] by
      [r], in that output's carrier ({!carrier}), rounded once to its
      destination's. Destinations follow [rs]: one for a [Monoid] or a
      [Logsumexp], the mean then the variance for [Moments], the extreme
      then its [int64] position for [Arg]. Each has the shape {!Shape.reduce}
      gives of [shape]. It answers [Wrong_dtype] where {!accepts} is [false]
      of a reduction and its output's dtype. *)

  val scan :
    ?carry:any ->
    Prog.t ->
    axis:int ->
    reduction * int ->
    dsts:any array ->
    load array ->
    answer
  (** [scan ?carry p ~axis (r, k) ~dsts loads] stores at each index the
      reduction [r] of output [k] of [p] over the indices along [axis] up to
      that index's, inclusive, in that output's carrier ({!carrier}), rounded
      once to its destination's. It is a running sum for [Monoid Sum], and
      the running population mean and variance for [Moments]. The result at
      an index is a function of the terms up to it alone, in the sense that
      the scan of a prefix of the axis is the prefix of the scan, bit for
      bit. A scan's last result need not equal the reduction of the same
      terms. Destinations are as {!reduce}'s, each of the loop's shape
      ({!Shape.scan}). It answers [Wrong_dtype] where {!accepts} is [false]
      of [r] and output [k]'s dtype.

      With [carry], of a [Monoid Sum], [Prod], [Max] or [Min], the scan of
      each slice continues from its element of [carry], an array of the
      destinations' shape with extent [1] along [axis], of the output's
      carrier ({!carrier}). Each result is then that element combined with
      the terms up to it, and a float or complex sum's or product's is
      within the bound of a reduction of those terms and the element
      ({!section-reductions}). An integer's or a boolean's scan, and a
      [Max]'s or a [Min]'s, is exact, so the scan of a slice's terms from
      any term, with the result before it, in the carrier, as the carry, has
      the bits of the scan of the whole slice from that term on, NaN results
      being any NaN. A NaN carry makes every result a NaN. It answers
      [Shape_mismatch] for a carry of any other reduction or of another
      shape, and [Wrong_dtype] for one of another dtype. *)

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
      element, its bits unchanged, a signalling NaN's included. Where the
      updates [u1], …, [un], in C order of their indices, land on [x], it
      is:
      - for [Set], [un];
      - for [Add], [+0 + x + u1 + … + un] by {!Prog.Add}, a narrow float's sum
        computed in float32 and rounded once. A sum has the same bits on
        every run, and each kernel library states how it associates it. For
        finite operands, barring overflow, a float sum is within
        [γ(n + 1) (|x| + Σ |ui|)] of the exact one before a narrow float's
        rounding, with [γ] as {!section-reductions} defines it and [u] the
        unit roundoff of float32 for a narrow float, else of the dtype's
        parts;
      - for [Max] and [Min], {!Prog.Maximum} and {!Prog.Minimum} of [x], [u1],
        …, [un], a NaN where one of them is ({!section-reductions}).

      Under [unique], the caller must give each target at most one update;
      where two share one, the result there is unspecified. It answers
      [Wrong_dtype] for an [Add] of a dtype {!Prog.Add} does not take. *)

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
    (Nx_array.Move.range iarray * ('v, 's) a) array ->
    answer
  (** [assemble ~fill ~dst pieces] stores into [dst], at each index, the
      element at that index of the last piece whose region holds it, or [fill]
      where none does. Each piece has the shape of its region
      ({!Shape.assemble}). Bits are copied, NaN payloads included. *)

  val fold : pad -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [fold p ~dst x] stores the adjoint of a [Padded] load by [p] of [dst]'s
      shape, where [x] has the shape that load gives. [dst]'s element at an
      index is the sum by {!Prog.Add} of [+0] and the elements of [x] the
      load reads from that index, in the association its library states.
      An element's {e tap} is its index along the axes the load's windows
      append. A float sum of [n] elements is within [γ(n - 1) Σ|x|] of the
      exact one, barring overflow and underflow, with [γ] as
      {!section-reductions} defines it and [u] the dtype's unit roundoff,
      and each part of a complex sum is such a sum. It answers
      [Wrong_dtype] for a dtype {!Prog.Add} does not take. *)

  (** {1:contraction Contraction}

      A float or complex contraction's association, which fixes its bits,
      is its library's choice, from the dtypes, the device model and the
      call's extents, as a blocking chosen by the problem's size.
      A segmented call's offsets never enter it, so an output's bits do not
      depend on which groups the other indices are in. A window
      of [dst] computed alone, or a part of the contracted extent, keeps
      the bound {!contract} states, not the whole call's bits. *)

  val contract :
    batch:(int * int) iarray ->
    contracting:(int * int) iarray ->
    segments:index segments option ->
    acc:Nx_array.Dtype.any ->
    dst:('v, 's) a ->
    init:('v, 's) a option ->
    operand ->
    operand ->
    answer
  (** [contract ~batch ~contracting ~segments ~acc ~dst ~init a b]
      stores into [dst] the contraction of [a] with [b], of the shape
      {!Shape.contract} gives:
      {[
      dst[B, I, J] = init[B, I, J] + Σ_K a[B, I, K] · b[B, K, J]
      ]}
      computed in [acc], a dtype {!accumulates} takes, summed in the
      library's order and rounded once to [dst]'s dtype, each operand's
      elements times their scales, each group of [segments] as
      {!type-segments} states. Without [init] the sum starts from [+0]. An
      output with no products, where the contracted extent is [0] or, under
      [segments], where it is in no group or in a group of no index, is
      [init] as [acc] holds it, rounded into [dst], or [+0] without [init].
      No addition touches it, so an [init] of [-0] stays [-0]. Integers wrap
      in [acc]. A float or complex [acc] errs by at most twice its unit
      roundoff, its parts' for a complex one, per addition, barring
      underflow. A NaN result's sign and payload are unspecified, but for an
      output of no products.

      A scale multiplies, as the library states, each element, rounded once
      into [acc], or each product, or, where a scale block along the
      contracted extent is whole, its partial sum. The library also states
      the roundings its scaling adds to the bound, barring overflow of what
      it scales. A
      [Value] or code that is NaN makes its scale block's terms NaN. *)

  (** {1:fft Fourier transforms}

      Each transform runs along each of [axes], strictly increasing and not
      empty. A transform of no point is the empty sum. Each kernel library
      states its error bound. A NaN result's sign and payload are
      unspecified. *)

  val c2c :
    direction -> axes:int iarray -> dst:(Complex.t, 's) a -> (Complex.t, 's) a -> answer
  (** [c2c d ~axes ~dst x] stores the transform [d] of [x] into [dst]. *)

  val r2c : axes:int iarray -> dst:(Complex.t, 'c) a -> (float, 'r) a -> answer
  (** [r2c ~axes ~dst x] stores the [Forward] transform of [x], bins [0] to
      [n/2] along the last of [axes], [n] [x]'s extent there. For [n = 0] it
      is one bin of [+0]. [x] and [dst] are of the {!same_precision}. *)

  val c2r : axes:int iarray -> dst:(float, 'r) a -> (Complex.t, 'c) a -> answer
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
      operand and destination. Each matrix is computed on its own, and one
      that fails is NaN in its own results alone. [aᴴ] is the conjugate
      transpose. Routines take the {!full_precision} dtypes; a real result of a
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
      name one result. [qr]'s factors are then the same either way, and
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
      [pivots[j]] the row a scan of column [j] down from row [j] ends at. The
      scan starts at [j] and moves to a later row only where that row's
      magnitude, [|re| + |im|] on complex, is greater than the current one's.
      A NaN is never greater, and nothing is greater than a NaN. Row [i] of
      [L U] is row [perm[i]] of [a]. A zero pivot stays in [U] and its column
      of [L] is not divided. *)

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
