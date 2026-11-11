(** Kernels: what they compute and the contract they implement.

    A {e kernel} computes one of nx's operations on arrays of one device
    ({!Rig.t}). It reads its {e operands} and writes its results into its
    {e destinations}, arrays its caller allocates. Its other arguments, such
    as the axes it reduces, are its {e attributes}. A kernel takes its
    attributes, then its destinations, then its operands, but for a scan's
    optional carry, which comes first, and a segmented contraction's
    ids, which come with its segments.

    A {e kernel library} is a module of kernels that compute on the memory
    of some devices. A {e case} is an operation at some dtypes, attributes
    and extents, and a library {e declines} a case when it answers without
    computing it. Its [.mli] states what it computes on, which cases it
    declines and the algorithms and error bounds this contract leaves to it,
    then includes {!S}, so that its build checks it against the contract:
    {[
    (** Kernels for host memory. … *)

    include Nx_kernel.S
    ]}

    A {e program} ({!Prog}) computes from one element of each of its
    operands. A {e loop} runs a program at every index of one shape
    ({!S.section-loops}), reading each operand as a {e load}, the array with
    how the loop reads it ({!type-load}). A reduction, a scan, a scatter's
    addition, a {e fold} ({!S.fold}) and a contraction combine several
    values, the result's {e terms}, which the operation's doc numbers. A
    reduction or a scan combines terms in their dtype's {e carrier}
    ({!carrier}), a dtype that holds every element of theirs. A {e move}
    copies bits: a gather, a range gather, a scatter by [Set], a sort, an
    assembly, and a map of [Copy] or [Bitcast]. A {e routine} is a kernel of
    {!S.section-linalg}. The sections below define each operation's types
    and rules, then {!Shape}, the shapes of every kernel's destinations,
    then {!S}.

    A call {e fits} its kernel when its operands and destinations are as
    many as the kernel's doc states, of the dtypes it takes and of the
    shapes it and {!Shape} state, and its attributes are ones {!Shape}
    answers no [Error] for. nx checks that a call fits before it calls a
    kernel, so that an error names the arrays, then allocates the
    destinations. The kernel claims its destinations and operands for the
    call through the {e door}, the claim of arrays {!Nx_array} defines, and
    answers as {!Nx_array.type-answer} states: [Done], [Declined] for a
    case it does not compute, or a {e refusal}, which names what is wrong
    with an operand. Any case may be [Declined], within the bounds
    {!S.section-loops} states for loops. nx raises a refusal as
    {!Nx_array.refused} does, and after a decline raises
    [Invalid_argument] and computes nothing in the library's place.

    The caller must pass a call that fits. Where a kernel answers
    [Declined] or a refusal for a call that fits, it does so before any
    write or queued work. For one that does not, it answers a refusal or
    [Declined], never [Done], and reads and writes nothing outside its
    arrays; the library chooses which refusal, and the destinations'
    contents are unspecified. nx names a misfit with the refusal
    {!map_refusal}, {!reduce_refusal}, {!scan_refusal} or
    {!attention_refusal} gives, or [Shape_mismatch] where {!Shape} answers
    [Error].

    A failure that depends on a float's or a complex number's value is a
    NaN result. A kernel raises
    [Out_of_memory] if host memory runs out, {!Rig.Out_of_memory} if its
    device's does, and {!Rig.Lost} as the door does. After a raise, the
    destinations' contents are unspecified.

    Unless a function's doc says otherwise:
    - The caller must pass destinations that are C-contiguous, reach no
      element twice and are on the operands' device. Operands may be
      strided, broadcast or offset.
    - A destination shares no byte with an operand, except one it is
      identical to that the kernel reads only at the result's own index: a
      [Plain] load of [map], [contract]'s [init], [scatter]'s [into], or
      [assemble]'s first piece where its region is the whole result. A
      kernel reads such an operand at an index before writing there, so the
      result equals what a fresh destination would receive. The door
      refuses other sharing as [Overlapping]. It accepts a destination
      identical to an operand read at other indices, and the caller must
      not pass one; the result is then unspecified unless the library's
      doc states it.
    - A kernel is {e deterministic}. For the same operand values, shapes,
      layouts and dtypes and the same attributes, it gives the same bits on
      every run, on any number of threads, on devices of one model, for a
      host library one architecture.
    - A result whose bits this contract fixes, an integer, a boolean, an
      extreme, a position, a move's or a map of kinds {!Prog} defines
      exactly, depends only on the operand elements its own index reads. So
      a {e box} of the results' indices, a range of them along each axis,
      computed alone from the parts of the operands it reads, has the bits
      of the whole.
    - Only a move specifies a NaN result's bits. Every other kernel's NaN
      result follows the NaN rule ({!Nx_array.Dtype.section-stores}): its
      sign and payload are unspecified, and it is signalling only where an
      operand is a signalling NaN. *)

module Prog = Prog
(** Scalar programs: what a loop computes at each index. *)

type ('v, 's) a := ('v, 's) Nx_array.t
type index := (int64, Nx_array.Dtype.int64_elt) Nx_array.t

(** The type for reading the dtypes of a collection of arrays of type ['c]
    in place: [arrays c] arrays, [dtype c i] the dtype of the [i]th. A
    rule that decides a kernel's refusals, as {!attention_refusal}, reads
    its caller's arrays through one, which the caller defines once per
    collection type, so that checking a call copies nothing. *)
type 'c dtypes = {
  arrays : 'c -> int;
  dtype : 'c -> int -> Nx_array.Dtype.any;
}

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
    neighbours, and a negative [lo] or [hi] crops instead. Then it takes the
    windows of the padded operand that {!Nx_array.Move.Window}[ windows]
    takes. {!Shape.padded} states which paddings an operand takes. *)

(** The type for loads. A load is an operand of a loop together with how
    the loop reads it. *)
type load =
  | Plain : ('v, 's) a -> load  (** The operand through its layout. *)
  | Padded : { x : ('v, 's) a; fill : 'v; pad : pad } -> load
      (** [x] padded with [fill] as [pad] says. The loop's shape is the padded
          one. *)

val map_refusal :
  Prog.t ->
  rank:int ->
  dsts:'d dtypes ->
  'd ->
  loads:'l dtypes ->
  'l ->
  (Nx_array.answer * string) option
(** [map_refusal p ~rank ~dsts:dv dsts ~loads:lv loads] is
    [Some (r, why)] where {!S.map}'s program [p] over [loads] into [dsts],
    read by [lv] and [dv], on a loop of rank [rank], does not fit: [r] the
    refusal that names the misfit and [why] what does not fit, in words;
    [None] where all fit. It reads no shape but the rank: {!Shape.padded}
    gives a padded load's. *)

(** {1:reductions Reductions and scans}

    A reduction's terms are the values of one program output along the
    reduced axes, numbered from [0] in C order of those axes' indices.

    A float result is a NaN where a term it reduces is a NaN, or where its
    arithmetic makes one from numbers, as [+∞] plus [-∞] does. To every
    reduction but [Sum], a complex number with a NaN part is a NaN. A
    complex [Sum] adds each part as a float sum.

    A float [Sum] or [Prod] reduction adds or multiplies each output's terms
    in the {e association} its library states, the order and grouping of
    its operations, a narrow float's terms in [float32], and a complex
    [Prod] multiplies them by {!Prog.Mul}. Each result rounds once into
    its destination, a float as {!Nx_array.Dtype.of_float} rounds it. *)

type monoid = Sum | Prod | Max | Min
(** The type for monoids a reduction folds with. *)

type extreme = Max | Min
(** The type for extremes a reduction finds with their positions. *)

val carrier : Nx_array.Dtype.any -> Nx_array.Dtype.any
(** [carrier dt] is the dtype kernels reduce and scan terms of [dt] in, and
    the dtype of a scan's {e carry}, the state a scan continues from
    ({!S.scan}): [float32] for a float narrower than it, [int8] and [uint8]
    for [int4] and [uint4], [bool] for [bit], else [dt]. *)

(** The type for a float dtype, whatever its storage. *)
type any_float = Any_float : (float, 's) Nx_array.Dtype.t -> any_float
[@@unboxed]

val float_carrier : (float, 's) Nx_array.Dtype.t -> any_float
(** [float_carrier dt] is {!carrier} of the float [dt], the compute type
    in which a float kind of [dt] computes before it rounds once to [dt]
    ({!Prog}). *)

(** The type for what a reduction computes from its terms. *)
type reduction =
  | Monoid of monoid
      (** [Monoid Sum] adds the terms from [+0], as {!Prog.Add} does, so a
          sum of [-0] terms is [+0]. [Monoid Prod] multiplies them from the
          first, never by [1], which would change a complex term's signed
          zeros or make NaN parts from its infinite ones, and an empty
          product is [1]. Integers wrap. [Monoid Max] and [Monoid Min] are
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
    every dtype but booleans, [Max], [Min] and [Arg] every dtype, [Logsumexp]
    and [Moments] floats. *)

(** The type for reading a collection ['c] of reductions in place, as
    {!type-dtypes} reads arrays: [reductions c] reductions, the [i]th
    [kind c i] of the program's output [output c i]. *)
type 'c reductions = {
  reductions : 'c -> int;
  kind : 'c -> int -> reduction;
  output : 'c -> int -> int;
}

val reduce_refusal :
  Prog.t ->
  rank:int ->
  reductions:'r reductions ->
  'r ->
  dsts:'d dtypes ->
  'd ->
  loads:'l dtypes ->
  'l ->
  (Nx_array.answer * string) option
(** [reduce_refusal p ~rank ~reductions:rv rs ~dsts:dv dsts ~loads:lv loads]
    is as {!map_refusal} for {!S.reduce}'s reductions [rs] of [p], read by
    [rv]. It reads no shape but the rank: {!Shape.reduce} gives the
    destinations'. *)

val scan_refusal :
  Prog.t ->
  rank:int ->
  reduction ->
  output:int ->
  dsts:'d dtypes ->
  'd ->
  ?carry:Nx_array.Dtype.any ->
  loads:'l dtypes ->
  'l ->
  (Nx_array.answer * string) option
(** [scan_refusal p ~rank r ~output:k ~dsts:dv dsts ?carry ~loads:lv loads]
    is as {!map_refusal} for {!S.scan}'s scan [r] of output [k] of [p]
    from a carry of dtype [carry]. It reads no shape but the rank:
    {!Shape.carry} gives the carry's. *)

(** {2:bounds Error bounds}

    Within the bounds this contract and its library state, a float or
    complex result may differ between libraries, device models and layouts,
    and the same terms in a call of other extents may round differently.
    Let [u] be the unit roundoff of the dtype each addition or
    multiplication rounds to, its parts' for a complex one,
    [γ(k) = k u / (1 - k u)] and
    [β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉].

    A library adds the terms of a float reduction and the products of a
    contraction in {e blocks}: no running sum takes more than 256 terms
    before it is added into another. A term so passes at most 256
    additions in its block, at most the cost of 320 on a matrix unit, whose
    additions align 8 terms and a running sum to the largest and truncate,
    and at most [⌈n / 128⌉] more as the blocks' sums, and the partial sums
    of threads and kernels, combine. Barring overflow and underflow, and
    before the rounding into the destination, which adds its own, a float
    sum of [n] terms, a reduction's, is within [γ(β(n)) Σ|x|] of the exact
    sum, and a scan's or a fold's within [γ(n - 1) Σ|x|]. A float product
    is within a relative [γ(n - 1)], and a complex one within a normwise
    relative [γ(2n - 2)], where each {!Prog.Mul} errs by at most [2u]. A
    library may state sharper bounds.

    A float or complex contraction ({!S.contract}) sums each output's [K]
    products [a·b] from [+0] in blocks, then adds [init] to their sum
    ({!section-contraction}). Barring overflow and underflow, and before
    the rounding into the destination, an output is within
    [u|r| + (1 + u) e(K) Σ|a||b|] of the exact [r = init + Σ a·b], with
    [init] as the accumulator holds it, a complex one in modulus, and the
    [u|r|] term [0] without [init]. [e] is the library's {e error factor},
    a function of [K] that the library states and that may vary with the
    dtypes and the device model, [u] the accumulator's unit roundoff, its
    parts' for a complex one. It is at most [γ(β(K))] for a real
    accumulator and [√2 γ(β(2K))] for a complex one, each of whose parts
    sums [2K] terms. *)

(** {2:scans Scans}

    A float or complex [Sum] or [Prod] scan combines the terms up to each
    result by {!Prog.Add} or {!Prog.Mul} in the carrier ({!carrier}), in the
    association its library states, a product from its first term, so that
    its first result is that term. A [Logsumexp] or [Moments] scan's result
    has the special values of that reduction of the terms up to it
    ({!type-reduction}), and each kernel library states its algorithm and
    its bounds. Each result rounds once into its destination. A scan's
    bounds are on the result in the carrier, before the rounding into the
    destination, which adds its own. They are to first order in the
    carrier's unit roundoff [u], barring overflow.

    The result at an index is a function of the terms up to it alone, in
    the sense that the scan of a prefix of the axis is the prefix of the
    scan, bit for bit. A scan's last result need not equal the reduction of
    the same terms. A scan of integers or booleans, and a [Max] or [Min]
    scan, is exact, so scanning a slice from any term, with the carrier
    value of the result before it as the carry, gives the whole scan's
    results from that term on, bit for bit but for a NaN result, which may
    be any NaN.

    {3:network The odd/even network}

    The {e odd/even network} scans a slice by combining states, one per
    term. Neighbours [2j] and [2j + 1] combine into pairs, the pairs scan by
    the same network, result [0] is state [0], result [2j + 1] is the pairs'
    result [j], and result [2j], for [j > 0], is the pairs' result [j - 1]
    combined with state [2j]. The earlier state comes first in each
    combination, and result [i] passes through at most
    [D = max 0 (2⌈log₂ (i + 1)⌉ - 1)] of them. A library that scans by it
    states its bounds in [D]. *)

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
    pairs' indices. A product is fused into the addition that takes it. A
    complex contraction sums each part of an output as a float sum of
    products of parts. With [a = ar + i·ai] and [b = br + i·bi], its product
    [p]'s terms are numbered [2p] and [2p + 1], [ar·br] and [ai·(-bi)] in
    the real part, [ar·bi] and [ai·br] in the imaginary part.

    An output's products sum from [+0] in its library's association, and
    its [init] element, as the accumulator holds it, is then added to their
    sum in one addition, so that [init] does not sit in the running sum
    while the products add: an [init] of [-0] and products summing to [+0]
    give [+0]. The result rounds once into the destination. An output
    without products, where the contracted extent is [0] or its segments
    give it none, is its [init] element as the accumulator holds it,
    rounded into the destination, or [+0] without [init]. No addition
    touches it, so an [init] of [-0] stays [-0] and a NaN [init] keeps its
    bits. *)

val accumulates : ('v, 's) Nx_array.Dtype.t -> bool
(** [accumulates dt] is [true] iff a contraction computes in [dt]: every
    dtype but booleans and floats narrower than [float32]. *)

(** The type for a block scale of a contraction operand's array: an array of
    its rank, each extent dividing the operand's, [0] only along an empty
    axis. The operand's element at index [i] is in the {e scale block} [i]
    divided by the ratio of the extents, axis by axis. The kernel applies a
    scale by the operand's own index, whatever its layout. The caller must
    not pass the array's scale for a view that cuts an axis within its
    scale blocks. *)
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

(** The type for a contraction cut into groups by [ids], an [int64] array for
    {!S.contract} (its shape for {!Shape.contract}) of the batch extents then
    the segmented axis's. At each batch index, the segmented axis's index [r]
    is in group [ids.{r}] where that lies in [\[0, g)], and in no group
    otherwise. *)
type 'i segments =
  | Rows of { ids : 'i; a_axis : int; b_axis : int }
      (** [a]'s free axis [a_axis] is segmented; [b]'s free axis [b_axis], of
          extent [g], leaves the result. Index [r] along [a_axis] multiplies
          [b]'s index [ids.{r}] along [b_axis]; an output whose index along
          [a_axis] is in no group has no products. *)
  | Contracting of { ids : 'i; groups : int; pair : int }
      (** Contracting pair [pair] is segmented on both sides into [groups]
          groups, [g]. The result gains a leading axis of [groups], slice [e]
          summing group [e]'s indices alone, its products numbered by their
          order among the group's indices from [0], its extent the group's
          count; a group of no index has no products. *)

(** {1:fft Fourier transforms} *)

(** The type for a transform's direction. Along an axis of [n] points,
    [Forward] is [y[k] = Σ_j x[j] e^(-2πi jk/n)] and [Inverse] the same sum
    with [e^(+2πi jk/n)]. Neither divides by [n]. *)
type direction = Forward | Inverse

val same_precision :
  ('v, 's) Nx_array.Dtype.t -> ('w, 't) Nx_array.Dtype.t -> bool
(** [same_precision x y] is [true] iff [x] and [y] have one precision, each
    [float32] or [complex64], or each [float64] or [complex128]. A real
    transform, {!S.svd}, {!S.eigh} and {!S.eig} pair dtypes of one
    precision. *)

(** {1:matrices Matrices}

    A routine computes on each matrix of [a], its last two axes, of [m] rows
    and [n] columns, [k = min m n]. Each kernel library states its error
    bounds and the choices the routines leave open: [eig]'s order of
    eigenvalues; [qr]'s factors, unique only up to a unit factor per column
    of [q] and row of [r] where [r]'s diagonal has no zero, and otherwise
    from the first zero on up to an orthonormal completion; the vectors of
    [svd], [eigh] and [eig], unique only up to a unit factor where their
    value is simple and up to a unitary basis of its space where it
    repeats; and the columns complete factors add, any orthonormal
    completion. *)

(** The type for a square matrix's triangles, each with its diagonal. *)
type triangle = Lower | Upper

(** The type for how much of a matrix's unitary factors {!Shape.qr} and
    {!Shape.svd} shape: [k] columns of the left one and [k] rows of the
    right one, or the square ones, of [m] and [n]. A kernel reads which from
    its destinations' shapes. Where a destination's two shapes coincide, as
    [q]'s, [r]'s and [u]'s do for [m <= n] and [vh]'s for [m >= n], they
    name one result. [qr]'s factors are then the same either way, and [svd]
    reads the choice from its other factor. *)
type factors = Reduced | Complete

val full_precision : ('v, 's) Nx_array.Dtype.t -> bool
(** [full_precision dt] is [true] iff [dt] is [float32], [float64],
    [complex64] or [complex128], the dtypes the matrix routines,
    {!S.cholesky} to {!S.solve_triangular}, take. *)

(** {1:attention Attention}

    Attention reads query rows [q] [[lead…; hq; tq; dk]] against keys and
    values of [hk] key heads, [hk] dividing [hq], query head [h] reading key
    head [h / (hq / hk)]. Its loop is [[lead…; hq; tq; tk]], a {e pair} at
    each index: a query row and a key. It computes in an {e accumulator},
    [float32] or [float64]. A pair's {e score} is [s × q·k], [s] the scale
    rounded to the accumulator and [q·k] the sum of the row's and the key's
    [dk] products, each element widened to the accumulator, in an order its
    library states. Barring overflow and underflow, [q·k] is within
    [e Σ|qᵢkᵢ|] of the exact sum, [e] a factor its library states, at
    most [γ(2dk + 2)]. A program over the pair
    computes two outputs from its operand [0], the score, and its loads,
    operands [1] on, each of the loop's shape: output [0], a [bool], which
    reads no score, and output [1], in the accumulator. Its [Coord] nodes
    read the loop's axes: [rank - 2] the query row, [rank - 1] the key.

    A pair is {e visible} where its key is before its lead index's length,
    [tk] without lengths, where a paged key's table entry is in \[[0],
    [pages]), and where output [0] holds. A row's [lse] is the log-sum-exp
    of output [1] over its visible pairs, and its [o] the sum of their
    values weighted by [exp (x - lse)], [x] each one's output [1], a sum
    that starts from [+0], so a sum of zeros is [+0]. A row of no visible
    pair, or of only [-∞] outputs, has [o = +0] and [lse = -∞]. A visible
    pair's value counts where its weight underflows to [0]: a NaN there
    makes the row's [o] a NaN. A value whose key is past the row's length,
    or whose table entry is outside \[[0], [pages]), is never read.
    Whether a NaN or an infinity at another hidden pair reaches [o] is the
    library's to state.

    Barring overflow and underflow, at a row with a visible pair, let [L]
    be the exact log-sum-exp of its visible outputs [1], [b] a bound on
    their magnitudes, [u] the accumulator's unit roundoff and
    [u' = u / (1 - u)]. Each library states a bound [ε] on [|lse - L|].
    Weights are per element of [o]: a library may compute and round each
    element's weight for a pair apart. Each weight [w] it computes in the
    accumulator is [exp (x - L + δ)], with
    [|δ| ≤ (1 + u) ε + (2b + ln tk) u + 5u']. Where the values are
    [float16] or [bfloat16], it may apply [r(c w) / c] in place of [w],
    the quotient exact, [r] the rounding to nearest, ties to even, into
    their dtype, and [c] a factor it chooses with [c ≥ 1] and [c w ≤ 1].
    Before its rounding into its dtype, an element of [o] sums the values
    [vⱼ] of the row's visible pairs times the weights [wⱼ] it applies to
    them at that element, in an order its library states, within
    [e(tk) Σ|wⱼ||vⱼ|] of [Σ wⱼ vⱼ], [e] as for [q·k]. *)

(** The type for keys and values. [Dense] holds [k] [[lead…; hk; tk; dk]]
    and [v] [[lead…; hk; tk; dv]]. [Paged] holds pools [k]
    [[pages; hk; page; dk]] and [v] [[pages; hk; page; dv]] read through
    [table] [[lead…; max_pages]]: key [t] of lead index [b] is the pools'
    entry [table[b; t / page]] at [t mod page], and
    [tk = max_pages · page]. *)
type ('a, 'i) kv =
  | Dense of { k : 'a; v : 'a }
  | Paged of { k : 'a; v : 'a; table : 'i }

val attention_refusal :
  Prog.t ->
  q:('v, 's) Nx_array.Dtype.t ->
  rank:int ->
  o:('w, 'r) Nx_array.Dtype.t ->
  acc:(float, 'c) Nx_array.Dtype.t ->
  loads:'l dtypes ->
  'l ->
  (Nx_array.answer * string) option
(** [attention_refusal p ~q ~rank ~o ~acc ~loads:v loads] is as
    {!map_refusal} for {!S.val-attention}'s program [p] over [loads], read by
    [v], a query of dtype [q] and rank [rank], [o] of dtype [o] and [lse]
    of dtype [acc]. It reads no shape but the rank: {!Shape.val-attention}
    gives the shapes. *)

(** {1:shapes Shapes} *)

(** Shapes of destinations.

    Each function is the shapes of a kernel's destinations for operands of
    the shapes it takes, the destinations its caller allocates, or why the
    operands do not fit. A call of other shapes does not fit. Every shape
    it derives, a padded operand's and each destination's, is one a layout
    admits ({!Nx_array.Layout.max_numel}), and one past it is an [Error].
    An axis is checked against the rank it indexes, in \[[0], [rank]). *)
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

  val reduce : axes:int iarray -> reduction array -> t -> t result
  (** [reduce ~axes rs y] is the shape of each destination of reductions
      [rs] along [axes] of a loop of shape [y]: [y] without [axes], and [y]
      itself for no axis. It is an [Error] if [axes] are not strictly
      increasing axes of [y], or if a [Monoid Max], [Monoid Min] or [Arg]
      of [rs] has a destination element and no term. *)

  val scan : axis:int -> t -> t result
  (** [scan ~axis y] is the shape of a scan's destinations along [axis] of a
      loop of shape [y]: [y]. *)

  val carry : axis:int -> t -> t result
  (** [carry ~axis y] is the shape of a carry of a scan along [axis] of a
      loop of shape [y]: [y] with extent [1] along [axis], with [axis] as
      {!scan} takes it. *)

  (** {1:index Gathers, scatters and sorts} *)

  val gather : axis:int -> t -> t -> t result
  (** [gather ~axis idx x] is the shape of a gather of [x] at positions of
      shape [idx]: [idx], which has [x]'s rank and extents but along
      [axis]. *)

  val gather_ranges : size:int -> t -> t -> t -> t result
  (** [gather_ranges ~size starts ends x] is the shape of a range gather's
      destination: [x] with [size] elements along axis [a = rank starts - 1].
      It is an [Error] unless [starts] and [ends] have one shape of at least
      one axis, [x] has more than [a] axes and [starts]' extents before
      [a], and [size] is not negative. *)

  val scatter : axis:int -> t -> t -> t -> t result
  (** [scatter ~axis into idx updates] is the shape of a scatter into [into]:
      [into]. [idx] and [updates] have one shape, [into]'s but along
      [axis]. *)

  val sort : axis:int -> k:int option -> t -> t result
  (** [sort ~axis ~k x] is the shape of a sort's [values] and [positions]: [x]
      with [k] elements along [axis] with [Some k], [x] with [None]. It is an
      [Error] for a [k] past that axis's extent or negative. *)

  (** {1:numbering Numbering} *)

  val unique : size:int -> t -> (t * t * t) result
  (** [unique ~size keys] is the shapes of a numbering's [ids], of its
      [first] and [counts], and of its [count], for [keys] of [[n; w]]:
      [[n]], [[size]] and [[]]. It is an [Error] unless [keys] has rank 2
      and [size] is not negative. *)

  (** {1:assembly Assemblies} *)

  val assemble : t -> Nx_array.Move.range iarray array -> t array result
  (** [assemble y regions] is the shape of each piece of an assembly of
      shape [y]: its region's, the elements [regions.(i)] keep of each axis
      of [y], as {!Nx_array.Move.Slice} keeps them. A fold's operand has the
      shape {!S.fold} states. *)

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
      its operand ({!type-scale}) and the segments' ids have the shape
      {!type-segments} states, with [a_axis] and [b_axis] free axes, [pair]
      a contracting pair and [groups] at least [0]. *)

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

  val svd : factors -> t -> (t * t * t) result
  (** [svd f a] is the shapes of [s], [u] and [vh]: [[…; k]], then
      [[…; m; k]] and [[…; k; n]] under [Reduced], [[…; m; m]] and
      [[…; n; n]] under [Complete]. *)

  val eigh : t -> (t * t) result
  (** [eigh a] is the shapes of [w] and [v] of [a], square: [[…; n]] and
      [a], for {!S.eigh} and {!S.eig} alike. *)

  val solve_triangular : t -> t -> t result
  (** [solve_triangular a b] is [b], [[…; n; r]], for [a] square of [n] and
      [b] of [a]'s batch. *)

  (** {1:attention Attention} *)

  val attention :
    t -> (t, t) kv -> lengths:t option -> t array -> (t * t) result
  (** [attention q kv ~lengths loads] is the shapes of attention's [o] and
      [lse], [[lead…; hq; tq; dv]] and [[lead…; hq; tq]], for [q]
      [[lead…; hq; tq; dk]]. It is an [Error] unless [q] has at least three
      axes, [kv]'s arrays have the shapes {!type-kv} states, [hk] divides
      [hq] (zero heads divide zero), [lengths] are [[lead…]], each load is
      [[lead…; hq; tq; tk]] and [tk] is one a layout admits. *)
end

(** {1:contract The contract} *)

(** The type for kernel libraries. A library's kernels keep the rules stated
    above and those of their sections. Whether one declines depends only on
    the dtypes, the attributes, the device model and the extents of the
    axes along which one result element reads every index: a reduction's,
    a scan's, a sort's or a transform's axes, a numbering's keys, the
    contracted extent, a routine's matrices, a gather's or a scatter's
    axis, a range gather's runs, attention's keys and their two
    dimensions, and the axes a padded load or a fold pads or windows. Each
    library states whether several domains may call its kernels
    concurrently. *)
module type S = sig
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  (** [name] names the kernels in messages, as ["nx.cpu"]. *)

  val computes_on : Rig.t -> bool
  (** [computes_on d] is [true] iff these kernels compute on [d]'s memory. *)

  (** {1:loops Loops}

      A loop runs its program at every index of its destinations' shape. A
      reduction's destinations drop [axes], so it takes the loop's shape as
      [shape]. A loop reads load [i] as the program's operand [i]. Its loads
      are one per operand of the program, each of that operand's dtype, and
      each has the loop's shape once loaded, a [Plain] load its array's and
      a [Padded] one the shape {!Nx_kernel.Shape.padded} gives. A loop without
      loads runs over its shape alone. Its [Coord] nodes read axes of that
      shape, below its rank. A reduction or a scan computes in its program
      output's carrier ({!Nx_kernel.carrier}), in the association and within the
      bounds {!Nx_kernel.section-reductions}, {!Nx_kernel.section-bounds} and
      {!Nx_kernel.section-scans} state. Its destinations are of any dtype but an
      [Arg]'s position, an [int64], and each result rounds once into its own, as
      {!Nx_kernel.Prog.Cast} stores it. Each reduction is of an output of the
      program, of a dtype {!Nx_kernel.accepts} takes for it, and a [reduce] has
      at least one. {!Nx_kernel.map_refusal}, {!Nx_kernel.reduce_refusal} and
      {!Nx_kernel.scan_refusal} decide whether a loop fits; {!Nx_kernel.Shape}
      gives the shapes.

      A library declines a loop for its number of loads, program outputs or
      reductions only past a bound its device imposes, which its doc states,
      and computes a wider loop in parts itself where it can. It may decline
      a loop for its dtypes, its kinds or its other attributes, as its doc
      states. *)

  val map : Prog.t -> dsts:any iarray -> load iarray -> answer
  (** [map p ~dsts loads] stores [p]'s output [k] at each index into
      [dsts.(k)], of that output's dtype. It is the one elementwise kernel.
      One kind over operands comes as the program of one node that
      {!Nx_kernel.Prog.of_node} makes, which {!Nx_kernel.Prog.single} and
      {!Nx_kernel.Prog.is_single} recognise. *)

  val reduce :
    Prog.t ->
    shape:int iarray ->
    axes:int iarray ->
    (reduction * int) iarray ->
    dsts:any iarray ->
    load iarray ->
    answer
  (** [reduce p ~shape ~axes rs ~dsts loads] reduces, for each [(r, k)] of
      [rs], output [k] of [p] over the loop of shape [shape] along [axes] by
      [r]. Its destinations follow [rs]: one for a [Monoid] or a
      [Logsumexp], the mean then the variance for [Moments], the extreme
      then its [int64] position for [Arg]. Each has the shape
      {!Nx_kernel.Shape.reduce} gives of [shape]. *)

  val scan :
    ?carry:any ->
    Prog.t ->
    axis:int ->
    reduction * int ->
    dsts:any iarray ->
    load iarray ->
    answer
  (** [scan ?carry p ~axis (r, k) ~dsts loads] stores at each index the
      reduction [r] of output [k] of [p] over the indices along [axis] up to
      that index's, inclusive ({!Nx_kernel.section-scans}). Its destinations are
      as {!reduce}'s, each of the loop's shape ({!Nx_kernel.Shape.scan}).

      With [carry], an array of the shape {!Nx_kernel.Shape.carry} gives, of the
      output's carrier, a [Monoid] scan of each slice
      continues from the slice's element of [carry]. Each result is that
      element combined with the terms up to it, and a float or complex one
      is within the bound of a reduction of the element and those terms
      ({!Nx_kernel.section-bounds}). A NaN carry makes every result a NaN. Only
      a [Monoid] scan takes a carry. *)

  (** {1:index Gathers and scatters}

      A gather reads, and a scatter writes, at positions along [axis] that
      an [int64] array holds, one per element. A gather and a scatter by
      [Set] are moves. Where the updates [u1], …, [un], in C order of their
      indices, land on one element [x], a scatter stores there:
      - for [Set], [un];
      - for [Add], the sum by {!Nx_kernel.Prog.Add} of [x] and [u1] to [un],
        with no [+0] before [x], so that a [-0] target and [-0] updates sum to
        [-0], in the association its library states, a narrow float's sum
        computed in [float32] and rounded once. A sum has the same bits on
        every run. For finite operands, barring overflow, a float sum is
        within [γ(n + 1) (|x| + Σ |ui|)] of the exact one before a narrow
        float's rounding, with [γ] as {!Nx_kernel.section-bounds} defines it and
        [u] the unit roundoff of [float32] for a narrow float, else of the
        dtype's parts;
      - for [Max] and [Min], {!Nx_kernel.Prog.Maximum} and
        {!Nx_kernel.Prog.Minimum} of [x], [u1], …, [un], a NaN where one of them
        is ({!Nx_kernel.section-reductions}). *)

  val gather : axis:int -> dst:('v, 's) a -> index -> ('v, 's) a -> answer
  (** [gather ~axis ~dst idx x] stores into [dst] at each index [j] [x]'s
      element at [j] with axis [axis] replaced by [idx]'s element at [j]. A
      position outside \[[0], [d]), [d] [x]'s extent along [axis], reads the
      element of zero bits: [+0], [false], [0 + 0i]. *)

  val gather_ranges :
    dst:('v, 's) a -> starts:index -> ends:index -> ('v, 's) a -> answer
  (** [gather_ranges ~dst ~starts ~ends x] packs runs of [x] along its axis
      [a = rank starts - 1] into [dst]. [starts] and [ends] are
      [[lead…; k]], [x] is [[lead…; n; cells…]] and [dst]
      [[lead…; size; cells…]] ({!Nx_kernel.Shape.gather_ranges}). At each lead
      index, position [p] of [dst] along [a] belongs to the first run [j] whose
      end [ends.{j}] exceeds [p], and reads [x] at [starts.{j} + p - e], [e]
      being [ends.{j - 1}], or [0] for [j = 0], computed in [int64] modulo
      [2{^64}]: it reads what the exact sum would, but where [starts.{j}]
      and [e] lie near opposite extremes of [int64]. A position past every
      end, or outside \[[0], [n]), reads the element of zero bits. So for
      [ends] the running sums of lengths that are not negative, run [j]
      copies its [ends.{j} - e] elements from [starts.{j}], cut at [size].
      It is a move: [dst] is {!gather} of [x] at these positions. *)

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
      [into] with [updates] combined by [c] at the positions [idx] holds.
      Update [j] targets [into]'s element at [j] with axis [axis] replaced
      by [idx]'s element at [j]; a position outside \[[0], [d]), [d]
      [into]'s extent along [axis], drops it. Where no update lands, the
      result is [into]'s element, its bits unchanged, a signalling NaN's
      included.

      Under [unique], the caller promises each target at most one update,
      which a library may rely on: where two share one, the result there is
      unspecified. [Add] takes the dtypes {!Nx_kernel.Prog.Add} takes. *)

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
      [x] along [axis], it keeps the first elements of the sorted order. The
      order is the one {!Nx_kernel.Prog}'s domains state, [-0] below [+0], with
      every NaN, and every complex number with a NaN part, above [+∞] and equal
      to each other: [-∞ < … < -0 < +0 < … < +∞ < NaN]. [descending] reverses
      the order and keeps equal elements in increasing position. *)

  (** {1:numbering Numbering} *)

  val unique :
    ids:index -> first:index -> counts:index -> count:index -> index -> answer
  (** [unique ~ids ~first ~counts ~count keys] numbers the rows of [keys], an
      [[n; w]] array whose elements are read as unsigned 64-bit words: two
      rows are one key iff their words are equal, and a [w] of [0] makes
      every row one key. Keys are numbered from [0] in order of their first
      rows, [k] of them. [ids], of [n] entries, gets each row's key.
      [first] and [counts] have one length [m]: entry [g] below [min m k]
      gets key [g]'s first row and its number of rows, and entries from [k]
      on get [n] and [0]. [count], 0-d, gets [k]. *)

  (** {1:assembly Assemblies and folds}

      An assembly is a move. A fold is the adjoint of a [Padded] load: where
      the load reads one element of its operand into several indices of the
      loop, the fold adds those indices' elements into that element's
      place. An element's {e tap} is its index along the axes the load's
      windows append, and a fold's terms are numbered in C order of their
      taps. *)

  val assemble :
    fill:'v ->
    dst:('v, 's) a ->
    (Nx_array.Move.range iarray * ('v, 's) a) iarray ->
    answer
  (** [assemble ~fill ~dst pieces] stores into [dst], at each index, the
      element at that index of the last piece whose region holds it, or [fill]
      where none does. Each piece has the shape of its region
      ({!Nx_kernel.Shape.assemble}). Bits are copied, NaN payloads included. *)

  val fold : pad -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [fold p ~dst x] stores into [dst] the adjoint of a [Padded] load by
      [p] of [dst]'s shape, where [x] has the shape that load gives.
      [dst]'s element at an index is the sum by {!Nx_kernel.Prog.Add} of its
      terms, the elements of [x] the load reads from that index, with no [+0]
      before the first, so that a single [-0] keeps its sign, and [+0]
      where it reads none; in the association its library states and
      within the bound {!Nx_kernel.section-bounds} states, each part of a
      complex sum such a sum. The library states whether a narrow float's sum
      rounds at each addition or once from [float32]. It takes the dtypes
      {!Nx_kernel.Prog.Add} takes. *)

  (** {1:contraction Contraction}

      A float or complex contraction's association, which fixes its bits,
      is its library's choice, from the dtypes, the device model and the
      call's extents, as a blocking chosen by the problem's size. A
      segmented call's ids never enter it, so an output's bits do not
      depend on which groups the other indices are in. A box of [dst]'s
      indices computed alone, or a part of the contracted extent, keeps the
      bound {!Nx_kernel.section-bounds} states; its bits may differ from the
      whole call's. The library states which a scale ({!Nx_kernel.type-scale})
      multiplies: each element, rounded once into the accumulator; each product;
      or the partial sum of a scale block that is whole along the contracted
      extent. It also states the roundings its scaling adds to the bound,
      barring overflow of what it scales. A [Value] or code that is NaN
      makes its scale block's terms NaN. *)

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
      {!Nx_kernel.Shape.contract} gives:
      {[
      dst[B, I, J] = init[B, I, J] + Σ_K a[B, I, K] · b[B, K, J]
      ]}
      computed in [acc], summed in the library's order and rounded once
      into [dst], each operand's elements times their scales, each group of
      [segments] as {!Nx_kernel.type-segments} states. [init] has [dst]'s
      shape and is added to the products' sum, which starts from [+0], as
      {!Nx_kernel.section-contraction} states, as is an output without
      products. Integers wrap in [acc].
      [a], [b], [init] and [dst] may be of any dtypes: an element enters
      [acc], and the sum enters [dst], as {!Nx_kernel.Prog.Cast} stores it. It
      takes an [acc] that {!Nx_kernel.accumulates} accepts. *)

  (** {1:fft Fourier transforms}

      Each transform runs along each of [axes], strictly increasing and not
      empty. A transform of zero points is the empty sum. Each kernel library
      states its error bound. *)

  val c2c :
    direction -> axes:int iarray -> dst:(Complex.t, 's) a -> (Complex.t, 's) a -> answer
  (** [c2c d ~axes ~dst x] stores the transform [d] of [x] into [dst]. *)

  val r2c : axes:int iarray -> dst:(Complex.t, 'c) a -> (float, 'r) a -> answer
  (** [r2c ~axes ~dst x] stores the [Forward] transform of [x], bins [0] to
      [n/2] along the last of [axes], [n] [x]'s extent there. For [n = 0] it
      is one bin of [+0]. [x] and [dst] are of one precision
      ({!Nx_kernel.same_precision}). *)

  val c2r : axes:int iarray -> dst:(float, 'r) a -> (Complex.t, 'c) a -> answer
  (** [c2r ~axes ~dst x] reads [n/2 + 1] bins [X] along the last of [axes], [n]
      [dst]'s extent there. It is the [Inverse] transform along each of [axes]
      but the last, then along the last the real parts of the [Inverse]
      transform of [X'], [X'[k]] [X[k]] for [k <= n/2] and [conj X[n - k]]
      above. The imaginary parts of bin [0] and, for an even [n], of bin [n/2]
      never contribute. It is this linear map for every operand, Hermitian or
      not. [dst] and [x] are of one precision ({!Nx_kernel.same_precision}). *)

  (** {1:linalg Matrices}

      Each routine computes on each matrix of [a]
      ({!Nx_kernel.section-matrices}), takes the {!Nx_kernel.full_precision}
      dtypes and leaves open the choices {!Nx_kernel.section-matrices} lists.
      [aᴴ] is the conjugate transpose. The routine's other axes, its batch, lead
      every operand and destination. Each matrix is computed on its own, and one
      that fails is NaN in its own results alone. [svd]'s singular values and
      [eigh]'s eigenvalues are floats, and [eig]'s results complex numbers, of
      [a]'s precision ({!Nx_kernel.same_precision}), for a real or complex [a].
      [qr] and [svd] compute reduced or complete factors as their destinations'
      shapes state ({!Nx_kernel.type-factors}). *)

  val cholesky : triangle -> dst:('v, 's) a -> ('v, 's) a -> answer
  (** [cholesky t ~dst a] stores [L] with [a = L Lᴴ] under [Lower], or
      [U = Lᴴ] under [Upper], [+0] in the other triangle and its diagonal real
      and positive. It reads [a]'s lower triangle and the real parts of its
      diagonal alone, whatever [t]. A matrix whose factorisation meets a
      pivot that is not positive, or NaN, is NaN. *)

  val lu : lu:('v, 's) a -> pivots:index -> perm:index -> ('v, 's) a -> answer
  (** [lu ~lu ~pivots ~perm a] stores into [lu] the unit lower triangular [L]
      below its diagonal, its ones unstored, and the upper triangular [U] on
      and above it. At step [j], rows [j] and [pivots[j]] were interchanged,
      [pivots[j]] the row a search of column [j] down from row [j] ends at.
      The search starts at [j] and moves to a later row only where that row's
      magnitude, [|re| + |im|] on complex, is greater than the current one's.
      A NaN is never greater, and nothing is greater than a NaN. Row [i] of
      [L U] is row [perm[i]] of [a]. A zero pivot stays in [U] and its column
      of [L] is not divided. *)

  val qr : q:('v, 's) a -> r:('v, 's) a -> ('v, 's) a -> answer
  (** [qr ~q ~r a] stores [q] with orthonormal columns and [r], upper
      triangular with [+0] below its diagonal, [a = q r]. *)

  val svd :
    s:(float, 'r) a ->
    ?vectors:(u:('v, 's) a * vh:('v, 's) a) ->
    ('v, 's) a ->
    answer
  (** [svd ~s ~vectors:(~u, ~vh) a] stores [a = u diag s vh], [s] descending
      and not negative, a zero one [+0]. Without [vectors] it stores [s]
      alone. A matrix that holds NaN or an infinity, or on which the
      iteration does not converge, is NaN in every result. [s] is of [a]'s
      precision ({!Nx_kernel.same_precision}). *)

  val eigh : w:(float, 'r) a -> ?v:('v, 's) a -> ('v, 's) a -> answer
  (** [eigh ~w ~v a] stores the eigenvalues of [a], square and Hermitian as
      its lower triangle and the real parts of its diagonal name it, which
      alone are read, ascending into [w], and orthonormal columns [v] with
      [a v = v diag w]. Without [v] it stores [w] alone. NaN as for [svd],
      over what is read. [w] is of [a]'s precision
      ({!Nx_kernel.same_precision}). *)

  val eig :
    w:(Complex.t, 'c) a -> ?v:(Complex.t, 'c) a -> ('v, 's) a -> answer
  (** [eig ~w ~v a] stores the eigenvalues of [a], square, into [w], and
      columns [v] of unit norm with [a v = v diag w]. Without [v] it stores
      [w] alone. NaN as for [svd]. [w] and [v] are of [a]'s precision
      ({!Nx_kernel.same_precision}). *)

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
      A zero on a diagonal it reads makes that matrix's [x] NaN. *)

  (** {1:attention Attention} *)

  val attention :
    Prog.t ->
    scale:float ->
    o:('w, 'r) a ->
    lse:(float, 'c) a ->
    ?lengths:index ->
    ('v, 's) a ->
    (('v, 's) a, index) kv ->
    any iarray ->
    answer
  (** [attention p ~scale ~o ~lse ?lengths q kv loads] stores the attention
      of [q]'s rows over [kv] ({!Nx_kernel.section-attention}), [p] the pair's
      program over its loads [loads], into [o] and [lse], whose dtype is
      the accumulator. [lengths], [[lead…]], bounds each lead index's
      keys. The scores, their maxima and sums, the weights and [lse]
      compute in the accumulator, save a weight applied through a rounding
      to the values' dtype ({!Nx_kernel.section-attention}), and each element of
      [o] rounds once into [o]'s dtype.

      [q], [k] and [v] are of one dtype, [float16], [bfloat16], [float32]
      or [float64], no wider than the accumulator, [o] is of the
      accumulator or of [q]'s dtype, and each load is of its operand's
      dtype. [p] has one operand more than [loads] and two outputs, its
      operand [0] and output [1] are in the accumulator, its output [0] is
      a [bool] that does not read operand [0], and its [Coord] nodes read
      axes below [q]'s rank. {!Nx_kernel.attention_refusal} decides whether
      these fit. Each library states its orders and its bounds. *)
end
