(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expansions: nx operations as simpler nx operations, applied through
    [apply]. An interpretation without a direct form for an operation
    computes it by its expansion ({!Eval.expand}, {!Nx.Prim.expand}), and
    {!Partial} composes the parts of a reduction split over devices from
    them.

    A map of several nodes expands into one-node maps, one per node in order,
    each over the values of the nodes it reads; a constant or a coordinate is a
    creation of the map's shape. A reduction or a scan that is not plain, one
    monoid of its operand itself accumulated and rounded in the operand's dtype,
    expands into a map of each output it reduces into its accumulator, the
    output's carrier ({!Nx_kernel.carrier}), the plain loop, and a cast to the
    result's dtype. A sum of complex numbers sums their parts, read as floats
    along a last axis of two.

    [Logsumexp], [Moments] and [Arg] reduce through plain loops, in the
    accumulator for the first two and in the output's dtype for [Arg]. A
    [Logsumexp] is a [Max] [m], [m'] its finite elements and [0] elsewhere, a
    [Sum] of [exp (x - m')], an [exp] of less than [-87] (float32) or [-708]
    (float64) taken as [0], then [m' + log s], or [m] where it is a NaN; with
    no term, [log 0]: nx.cpu's two passes ({!Nx_cpu}), and their bits. A
    float narrower than its accumulator rounds into its own dtype by a cast,
    which keeps a NaN's sign and payload. [Moments] are the mean [Σ x / n],
    then two sums of
    [d = x - mean] and the variance [(Σ d² - Σ d (Σ d / n)) / n], [+0] where
    that is negative or [Σ d²] where that is [+∞], each the mean where it is
    a NaN. An [Arg] is
    the [Max] or [Min] [m], then a [Min] of the positions, numbered in C order
    of the reduced indices, whose term has [m]'s bits or, where [m] is a NaN, is
    a NaN. Where a float sum of [n] terms is within [γ(n - 1) Σ|x|] of the exact
    sum ({!Nx_kernel.section-reductions}), and barring overflow and
    underflow, a [Logsumexp] [r] of [n] terms whose maximum is finite is within
    [γ(2n) + u|r| + ε] of the exact one, [ε] the error [Exp] and [Log] add; the
    mean within [γ(n) M], [M = Σ|x| / n], and the variance [V] within
    [(γ(n + 4) + 2γ(n)²) V + 2γ(n + 6) γ(n) M √V + 4γ(n + 4) γ(n)² M²], for
    [n] exact in the accumulator. A scan of [Logsumexp], [Moments] or
    [Arg], the first two in the accumulator, combines its terms' states by
    {!Pairwise.scan}: output [i] through at most
    [D_i = max(0, 2⌈log₂(i+1)⌉ - 1)] combinations, the earlier state first,
    O(n) along an axis of [n]. [Logsumexp] and [Moments] scan as
    {!Nx_kernel.section-scans} states. [Arg] combines pairs of an extreme and
    its position, keeping the earlier pair unless the later extreme is past
    it in [Max]'s or [Min]'s order: exact. A NaN result of these scans,
    carried through the combinations with its bits, is then quieted, as the
    kernels give it.

    A one-node map at a dtype other than a base one ([float32], [float64], the
    8- to 64-bit integers and [bool]), but a cast, a bitcast or a copy, computes
    at its dtypes' accumulators, which hold their values exactly, and rounds
    once to its dtype. A selection, a constant and a copy move bits: a float of
    8 or 16 bits computes as the unsigned integer of its width, over the same
    bits, and a sub-byte dtype in its byte-wide accumulator, which holds each of
    its codes and gives it back; a [Copy] is the map of one copy. An assembly
    expands into a fill of its flat result, then per piece in order a scatter of
    the piece's elements at their flat positions, a map of coordinates: O(n) per
    piece. A gather, a scatter or a sort of a sub-byte dtype casts its values to
    their accumulator, computes there and casts the result back once; an
    integer's [Add] wraps there to the same bits as at its own dtype. Any other
    scatter whose targets may repeat is a loop of one unique scatter per
    position along its axis, in order, each of the target so far and of that
    position's indices and updates, gathered at a broadcast of the trip's index:
    O(m |into|) for m positions. A sum associates left to right; a narrow
    float's sums in its accumulator, rounded once, and a [Set] of [true] at the
    same targets keeps the bits of each target no update reaches: twice the
    scatters.

    A contraction whose accumulator is not its output's dtype first runs as a
    contraction into the accumulator, then casts to the output. Any other
    first maps each scaled operand's elements, cast to the accumulator, times
    their block's scale read into it, rounded once: each axis split into its
    blocks, the scale broadcast within them, a code [e] read as float32's
    [2^(e - 127)] built from its bits and NaN at [255]. It then expands into
    a map of each pair of elements multiplied in the accumulator, a sum over
    the contracted axes, and the [init] added. Each index of a segmented
    axis takes its group id, a [Max] over the groups of [e] where
    [offsets.(e)] is at most the index, or [-1] outside [offsets.(0)] to
    [offsets.(g)]. [Rows] gathers each row's group of [b] by id into an
    axis of the rows, contracts it with [a]'s rows as a batch pair, and
    selects [init] or [+0] for a row in no group. [Contracting] sums each
    group's slice over the products whose id is its own, selecting [+0] for
    the others, so that no NaN or infinity of another group enters it, and
    selects [init] for a group of no index. An output of no products is its
    [init], or [+0] without one: adding the [+0] of an empty sum would turn
    an [init] of [-0] into [+0]. Only there is a zero's sign the kernels'.
    Where an output has products, their sum starts from [+0] and [init] is
    added last, so a zero result's sign may differ from a kernel's, within
    the bound {!Nx_kernel.S.contract} states: from an [init] of [-0] over
    products all [-0], nx.cpu's [Chain] order gives [-0] and the expansion
    [+0].

    A sort of another dtype sorts positions by its order
    keys, int64 values of [b = 63 - ⌈log₂ n⌉] bits along an axis of [n]
    elements, [⌈w / b⌉] for an integer or a float of [w] bits and per part for
    a complex number, the least significant first: each pass a bitonic network
    over the key packed above its position in an int64, along the axis padded
    to a power of two [N], then the elements gathered once: O(K N log² N) for
    K keys. A map with a padded load reads it
    as an assembly of its operand padded, each element it keeps at its padded
    position and the fill elsewhere, viewed through its windows: O(padded
    elements) per load; a reduction or a scan with a padded load expands into
    such a map. A fold is maps, each adding a run of at most 14 taps in their C
    order to the running sum, the first from [+0]: a tap is the fold's operand
    at its window indices, read by the padded load that is the adjoint of the
    fold's along each axis, so each element sums as {!Nx_kernel.S.fold}
    states: O(taps / 14) passes over the result. A transform computes on float
    pairs, a complex value's real parts then its imaginary parts, one axis after
    another: along an axis of a power of two [n], Stockham's autosort, [log₂ n]
    maps; along another, Bluestein's chirp, a convolution by three such
    transforms at the power of two [M >= 2n - 1]. [R2c] gives bins [0] to [n/2]
    of its operand with [+0] imaginary parts, and [C2r] completes its last
    axis's bins from their conjugates and keeps the real parts. With [Cos] and
    [Sin] within two units in the last place, a transform [y] of [N] points of
    [x] is within [‖ŷ - y‖₂ <= 20 u P √N ‖x‖₂], [P] the sum over its axes of
    [log₂ n] for a power of two and [3 log₂ M + 2] otherwise, [u] the parts'
    unit roundoff, and a [C2r]'s [x] its completed bins.

    A matrix routine computes each matrix of its batch by maps, gathers,
    reductions, sorts and contractions, its steps a loop whose every trip
    applies the same operations at full size, masked by the coordinates
    against the trip's index. Its bounds hold where no result overflows or
    underflows, to first order where they say so, with [u] the unit roundoff
    of the matrix's parts and [ν] the most one operation errs by: [2u] on
    floats, a contraction's addition ({!Nx_kernel.S.contract}), and [4u] on
    complex numbers, a quotient's ([nx_kinds.h]). [γ(k) = k ν / (1 - k ν)]
    bounds [k] such errors compounded.
    - [Cholesky] takes dpotf2's columns: [L Lᴴ = A + ΔA] with
      [|ΔA| <= γ(n + 1) |L| |Lᴴ|]. O(n³).
    - [Solve_triangular] substitutes row by row, an upper triangle as the
      lower one of its rows and columns reversed: [(T + ΔT) x = b] with
      [|ΔT| <= γ(n) |T|]. O(n² r).
    - [Lu] takes dgetf2's right-looking steps with {!Nx_kernel.S.lu}'s
      pivots: [L U = P A + ΔA] with [|ΔA| <= γ(k) |L| |U|]. O(m n k).
    - [Qr] takes dgeqr2's Householder reflectors as zlarfg makes them, each
      from its column over [μ], a power of two in the column's greatest
      binade, as zlarfg rescales: [τ] and [v] are the same of [x / μ] as of
      [x], and [β] is [μ β']. It then accumulates [Q]'s first [w] columns
      from the identity's, the last reflector first, as dorg2r does:
      O(m n k). A reflector applied to a vector [x] errs by [τ ‖x‖₂],
      [τ = ν + 2γ(8m + 24)], and [ε = (1 + τ){^k} - 1] compounds [k] of
      them (Yang, Fox and Sanders, SISC 2021, (3.9)-(3.10) and Lemma 3.3).
      There is a unitary [Q̂], the product of the exact reflectors of the
      computed vectors, with [Q̂ R = A + ΔA] and [‖Δa_j‖₂ <= ε ‖a_j‖₂], and
      the computed [Q]'s column [j] is within [ε] of [Q̂]'s, the same
      reflectors applied to the identity's (their Theorem 3.4). So
      [Qᴴ Q - I]'s elements are at most [2ε + ε²] and the computed
      product's column [j] is within [(1 + √w (1 + ε)) ε ‖a_j‖₂] of [a_j],
      on wide and complex matrices too.
    - [Eigh] and [Eigh_values] are two-sided Jacobi on the Hermitian
      matrix that [a]'s lower triangle and its diagonal's real parts name,
      scaled first by the power of two nearest the reciprocal of its
      greatest magnitude. A trip rotates the pairs of one round of a
      1-factorisation of the [n] indices into [R] rounds, [R = n - 1] for an
      even [n] and [n] for an odd one: the round of greatest weight
      [Σ |a_pq|²]. A rotation moves [2 |a_pq|²] of
      [off(A)²], the squared Frobenius norm of [A]'s off-diagonal part, to
      the diagonal and the rounds share [off(A)² / 2] out, so a trip takes
      at least [off(A)² / R]. [T] trips, the least with
      [(1 - 1/R){^T} <= u²], about [1.4 p R] for [u = 2{^-p}], each O(n²),
      leave [off(A) <= u ‖A‖_F] in exact arithmetic. A rotation applied to
      a vector errs by [ρ = 6u] on floats and [24u] on complex numbers, and a
      trip perturbs [A] by [3ρ ‖A‖_F] at most, its two sides and the pair it
      zeroes, so to first order an eigenvalue is within
      [(u + (3T + 6R) ρ) ‖A‖_F] of the exact one, [V]'s columns within
      [(1 + ρ){^T} - 1] of orthonormal ones and
      [‖A V - V diag w‖_F <= (u + (3T + 6R + 2√n T) ρ) ‖A‖_F].
    - [Svd] and [Svd_values] are one-sided Jacobi as dgesvj computes it, on
      [a], or [aᴴ] where [a] is wide, of [m'] rows and [k] columns, scaled
      as [Eigh]'s: [Qr]'s reflectors reduce it to its triangle [R], then 30
      sweeps of [R] rounds, dgesvj's, rotate the columns of [Rᴴ], as LAWN
      169's Corollary 2.5 and dgejsv take it, round [t mod R] at trip [t],
      each trip O(k²). A pair
      rotates where its columns' cosine [|y_pᴴ y_q| / (‖y_p‖ ‖y_q‖)] exceeds
      [tol]: dgesvj's, [√k u] for [Svd] and [k u] for [Svd_values], plus
      [2ρ + γ(k + 2)], the floor a rotated pair's computed cosine reaches
      here, [ρ] as for [Eigh]. Each column's norm is [Qr]'s; a column of a
      norm below the least normal number counts as zero and never rotates.
      As dgesvj, a matrix is converged after a sweep that rotates no pair;
      one whose last sweep still rotated is NaN. The values are the
      columns' norms, descending. With [ε_qr] [Qr]'s [ε] for [m'] rows and
      [ε_J = (1 + ρ){^30 R} - 1], to first order a singular value is within
      [((k - 1) tol + γ(k + 4)) σ + (ε_qr + ε_J) ‖A‖_F] of the exact one
      [σ]; for [A = B D] with [D] diagonal, each also relatively within
      [√k (ε_qr ‖B{^+}‖ + ε_J ‖R_c{^-1}‖)], [R_c] [R] with unit columns
      (Drmač and Veselić, LAWN 169, Proposition 2.4 and Corollary 2.5).
      With [Rᴴ V = Y], [R = V Σ Wᴴ], [W] [Y]'s columns over their norms
      completed by [Qr] of [k × k] to an orthonormal basis, each column by
      the sign of its [R]'s diagonal: the left vectors are the first
      reduction's [Q] times [V], the right ones [W], each within
      [ε_qr + ε_J] of orthonormal ones. To first order
      [‖u diag s vh - A‖_F <= √k (2k tol + 3ε_qr + ε_J + γ(k + 4)) ‖A‖_F].
    - [Eig] and [Eig_values] have no expansion: shifted QR has no bound on
      its iterations for a matrix that is not normal that a loop of static
      trips could afford.

    A routine fails as {!Nx_kernel.S} states, in the matrices that fail
    alone. Every other operation has no expansion. *)

val quieted :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [quieted apply ~by x] is [x] with each NaN of a float, or NaN part of a
    complex number, quieted, as the kernels return a NaN term: its quiet bit,
    the top bit of its fraction, set, its sign and the rest of its payload
    kept. Other dtypes are [x]. One map; a complex [x] is read as its parts,
    so it needs an axis below the greatest rank. *)

val sort :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  bits:int ->
  int ->
  bool ->
  int option ->
  ('v, 's, 'd) Value.t ->
  (('v, 's, 'd) Value.t * 'd Value.index) option
(** [sort apply ~by ~bits axis descending k x] is the expansion of
    [Sort { axis; descending; k; x }] by order keys of [bits] bits, for [bits]
    in \[[1], [63 - ⌈log₂ n⌉]\] along an axis of [n] elements, so that a key
    packed above its position fits an int64. {!run} sorts at the width [b]
    stated above. *)

val square_mod :
  int ->
  (int64, Nx_array.Dtype.int64_elt) Nx_kernel.Prog.E.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_kernel.Prog.E.t
(** [square_mod m t] is [t² mod m] at the int64 [t], exact for [t] in
    \[[0], [2{^63}]) and [m] in \[[1], [2{^54}]\]: the moduli [2n] of
    Bluestein's chirp along an axis of [n] elements. For [m] up to [2{^31}] it
    is [(t mod m)² mod m]; past it, a product by Horner's rule over digits of
    [t mod m] small enough that no step leaves int64. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  repeat:
    ('d.
    int ->
    ('d Value.index -> 'd Value.any list -> 'd Value.any list) ->
    'd Value.any list ->
    'd Value.any list) ->
  'r Value.prim ->
  'r option
(** [run apply ~by ~repeat op] is [Some r], [r] [op]'s expansion applied
    through [apply], or [None] for an operation that has no expansion, [Eig]
    and [Eig_values] among them. A loop of [n] trips runs as
    [repeat n step init], whose law {!Nx.Prim.expand} states. No result of
    [r] is one of [op]'s operands: one that would be is its copy. *)

val reduced :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  Nx_array.Layout.t ->
  int iarray ->
  Nx_kernel.Prog.t ->
  ('d, 'r) Value.reductions ->
  'd Value.load iarray ->
  'r option
(** [reduced apply ~by layout axes prog rs loads] is {!run} of
    [Value.Reduce { layout; axes; prog; reductions = rs; loads }], whose
    expansion has no loop. *)

