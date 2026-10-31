(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk
module D = Nx_array.Dtype
module P = Nx.Prim

let unlowered what =
  invalid_arg (Printf.sprintf "Rune.Lower.op: %s is not lowered" what)

let dtype = Ops.dtype
let ints l = List.map (fun n -> Ops.Int n) l
let zero u = Shape.const_like u (`Float 0.)
let float u x = Shape.const_like u (`Float x)
let transpose u = Shape.transpose u (-2) (-1)
let abs x = Lower_real.unary Abs x
let fdiv x y = Lower_real.binary Fdiv x (Shape.expand y (Shape.shape x))

(* The batch axes of the matrices of [u], their rows and their columns. *)
let matrix u =
  let dims = Shape.max_shape u in
  let r = List.length dims in
  ( List.filteri (fun i _ -> i < r - 2) dims,
    List.nth dims (r - 2),
    List.nth dims (r - 1) )

(* [block u rows cols] is the rows and columns of the matrices of [u] in the
   bounds [rows] and [cols], each all of them if [None]. *)
let block u rows cols =
  let r = Shape.ndim u in
  let bound = Option.map (fun (lo, hi) -> (Ops.Int lo, Ops.Int hi)) in
  Shape.shrink u
    (List.init r (fun i ->
         if i = r - 2 then bound rows
         else if i = r - 1 then bound cols
         else None))

(* [part u axis lo hi] is the elements [lo] to [hi], excluded, of [axis]. *)
let part u axis lo hi =
  Shape.shrink u
    (List.init (Shape.ndim u) (fun d ->
         if d = axis then Some (Ops.Int lo, Ops.Int hi) else None))

(* The diagonal of the matrices of [u]: the first column of their elements laid
   out in rows of one more element. *)
let diagonal u =
  let batch, n, _ = matrix u in
  let flat = Shape.reshape u (ints (batch @ [ n * n ])) in
  let padded =
    Shape.pad flat
      (List.map (fun _ -> None) batch @ [ Some (Ops.Int 0, Ops.Int n) ])
  in
  Shape.reshape
    (block
       (Shape.reshape padded (ints (batch @ [ n; n + 1 ])))
       None
       (Some (0, 1)))
    (ints (batch @ [ n ]))

(* The row and the column of each element of a matrix of [m] rows and [n]
   columns. *)
let row_index m = Shape.reshape (Shape.arange ~dtype:Int32 m) (ints [ m; 1 ])
let column_index n = Shape.reshape (Shape.arange ~dtype:Int32 n) (ints [ 1; n ])
let eye dt n m =
  Ops.cast (Ops.eq (row_index n) (Shape.arange ~dtype:Int32 m)) dt

(* -1 below zero and 1 elsewhere: the sign of a nonzero element, and 1 for a
   zero one. *)
let direction x = Ops.where (Ops.lt x (zero x)) (float x (-1.)) (float x 1.)

(* Indexed reads

   [take x axis p] is the element of [x] at position [p] along [axis], at each
   index of [p], whose other extents are [x]'s or smaller: a read of [x]'s
   elements in C order at [Σ index · stride], which moves each element with its
   bits. A position outside the axis reads its nearest end; one arises only on
   a matrix whose results are NaN. *)
let take x axis p =
  let s = Shape.max_shape x and ps = Shape.max_shape p in
  let n = List.nth s axis in
  if n = 0 || List.mem 0 ps then
    Shape.expand (Ops.const ~dtype:(dtype x) (`Int Bigint.zero)) (ints ps)
  else
    let r = List.length s in
    let stride d = List.fold_left ( * ) 1 (List.filteri (fun e _ -> e > d) s) in
    let coord d =
      if d = axis then
        let bound k = Shape.const_like p (`Int (Bigint.of_int k)) in
        let p = Ops.maximum (Ops.minimum p (bound (n - 1))) (bound 0) in
        Ops.cast p Weak_int
      else
        let k = List.nth ps d in
        Shape.expand
          (Shape.reshape
             (Shape.arange ~dtype:Weak_int k)
             (ints (List.init r (fun e -> if e = d then k else 1))))
          (ints ps)
    in
    let term d =
      match stride d with 1 -> coord d | k -> Ops.mul (coord d) (Ops.int k)
    in
    let at =
      List.fold_left (fun at d -> Ops.add at (term d)) (term 0)
        (List.init (r - 1) succ)
    in
    Ops.index (Shape.reshape x [ Ops.Int (List.fold_left ( * ) 1 s) ]) [ at ]

(* [at axis u i] is element [i] of [u] along [axis], one per matrix, the axis
   kept of extent one: [i] is a scalar or of [u]'s rank. *)
let at axis u i =
  let s = Shape.max_shape u in
  let one = List.mapi (fun d n -> if d = axis then 1 else n) s in
  let i =
    if Shape.ndim i = List.length s then i
    else Shape.reshape i (ints (List.map (fun _ -> 1) s))
  in
  take u axis (Shape.expand i (ints one))

(* [first_max axis u] is the position of the first greatest element of [u]
   along [axis], which holds no NaN, the axis kept of extent one: of the
   elements equal to the maximum, the one that counts down furthest from the
   axis's extent. *)
let first_max axis u =
  let r = Shape.ndim u and n = List.nth (Shape.max_shape u) axis in
  let keep op x = Shape.unsqueeze (Shape.rop x op [ axis ]) axis in
  let down =
    Shape.reshape
      (Shape.arange ~start:n ~step:(-1) ~dtype:Int32 0)
      (ints (List.init r (fun d -> if d = axis then n else 1)))
  in
  let hit = Ops.where (Ops.eq u (keep Op.Max u)) down (Ops.int 0) in
  Ops.sub (Ops.int n) (keep Op.Max (Ops.cast hit Int32))

(* [order ~descending x] is, along the last axis of [x], the position of each
   element of [x] in its stable sort: the [k]th is the [i] whose rank, the
   number of elements that sort before it, is [k]. An element sorts before
   another when it is less, greater under [descending], or equal and earlier.
   Its ranks and positions are exact integer sums. *)
let order ~descending x =
  let r = Shape.ndim x in
  let n = List.nth (Shape.max_shape x) (r - 1) in
  if n <= 1 then Shape.const_like ~dtype:Int32 x (`Int Bigint.zero)
  else
    let i = Shape.reshape (Shape.arange ~dtype:Int32 n) (ints [ n; 1 ]) in
    let j = Shape.arange ~dtype:Int32 n in
    let xi = Shape.unsqueeze x (-1) and xj = Shape.unsqueeze x (-2) in
    let ahead = if descending then Ops.gt xj xi else Ops.lt xj xi in
    let before =
      Ops.bitwise_or ahead (Ops.bitwise_and (Ops.eq xj xi) (Ops.lt j i))
    in
    let rank = Shape.rop (Ops.cast before Int32) Op.Add [ r ] in
    let at_k = Ops.eq (Shape.unsqueeze rank (-1)) j in
    let i = Shape.expand i [ Ops.Int n; Ops.Int n ] in
    Shape.rop (Ops.where at_k i (Ops.int 0)) Op.Add [ r - 1 ]

(* [by_order order x] is the columns of the matrices of [x] in [order]. *)
let by_order order x =
  take x (Shape.ndim x - 1)
    (Shape.expand (Shape.unsqueeze order (-2)) (Shape.shape x))

(* Failing matrices

   A matrix that breaks a routine's precondition on its values has results
   whose every element is NaN, as nx states. Whether each matrix keeps its
   results is a flag of its batch axes and two axes of one. *)

(* [defined ok u] is [u] where the flag [ok] holds, and NaN in every element of
   the matrices of [u] where it does not: of its vectors, when [u] has an axis
   fewer than [ok]. *)
let defined ok u =
  let ok =
    if Shape.ndim u < Shape.ndim ok then Shape.squeeze ~axis:(-1) ok else ok
  in
  Ops.where (Shape.expand ok (Shape.shape u)) u (float u Float.nan)

(* [over_matrix op u] is each matrix of [u] reduced by [op] to one element. *)
let over_matrix op u =
  let r = Shape.ndim u in
  Shape.unsqueeze (Shape.unsqueeze (Shape.rop u op [ r - 2; r - 1 ]) (-1)) (-1)

(* Whether every element of each matrix of [x] is finite: below infinity in
   magnitude, which NaN is not. *)
let finite x =
  let infinite =
    Ops.where (Ops.lt (abs x) (float x Float.infinity)) (zero x) (float x 1.)
  in
  let count = over_matrix Op.Add infinite in
  Ops.eq count (zero count)

(* Whether the elements off the diagonal of each matrix of [x] have a norm
   within [tol] times the norm of the matrix. The matrix is divided by its
   largest magnitude first, so that no square overflows. *)
let diagonal_within tol x =
  let _, m, n = matrix x in
  let largest = over_matrix Op.Max (abs x) in
  let x =
    fdiv x
      (Ops.where (Ops.gt largest (zero largest)) largest (float largest 1.))
  in
  let squares u = over_matrix Op.Add (Ops.mul u u) in
  let off =
    squares (Ops.where (Ops.eq (row_index m) (column_index n)) (zero x) x)
  in
  Ops.bitwise_or
    (Ops.lt off (Ops.mul (squares x) (float off (tol *. tol))))
    (Ops.eq off (zero off))

(* The [int32] exponent of the largest magnitude of each matrix of [u], of two
   axes of one: [e] with that magnitude in [[2^e, 2^(e + 1))], and [0] for a
   matrix of zeros or one that holds NaN or an infinity. *)
let exponent u =
  let largest = over_matrix Op.Max (abs u) in
  let usable =
    Ops.bitwise_and
      (Ops.gt largest (zero largest))
      (Ops.lt largest (float largest Float.infinity))
  in
  Lower_real.ilogb (Ops.where usable largest (float largest 1.))

let negate k = Ops.sub (Shape.const_like k (`Int Bigint.zero)) k

(* The unit roundoff of the float type [dt]. *)
let roundoff dt = Float.ldexp 1. (-(snd (Dtype.finfo dt) + 1))

(* [norm_roundoff l] is the unit roundoff times the Frobenius norm of each
   symmetric matrix whose lower triangle [l] holds, zeros above it: an element
   below the diagonal counts twice. [l]'s magnitudes are below 2, so that no
   square overflows.

   [eigh]'s floor is this absolute one, where [svd]'s is relative to each pair.
   A floor relative to the pair's diagonal elements vanishes with them in a
   cluster of zero eigenvalues, as of a rank-deficient matrix, which then keeps
   turning by arbitrary angles and converges a bit a sweep; the fixed count of
   sweeps would need [svd]'s extra one to cover it. A graded matrix's small
   eigenvalues are therefore accurate only to the interface's normwise
   bound. *)
let norm_roundoff l =
  let _, m, n = matrix l in
  let twice =
    Ops.where (Ops.eq (row_index m) (column_index n)) (float l 1.) (float l 2.)
  in
  let squares = over_matrix Op.Add (Ops.mul twice (Ops.mul l l)) in
  Ops.mul (Ops.sqrt squares) (float squares (roundoff (dtype l)))

(* Loops

   A routine repeats its step in a loop on the graph's device
   ({!Loop.repeat}). A loop of no trip, or of values with no element, runs
   nothing and needs none. *)

let repeat device n init step =
  let empty u = Shape.max_numel u = 0 in
  if n = 0 || List.for_all empty init then init
  else Loop.repeat device n init step

(* [dot a b] is the product of the matrices of [a] and [b]: each row of [a]
   against each column of [b] along a new last axis, summed. *)
let dot a b =
  let p =
    Ops.mul (Shape.unsqueeze a (-2)) (Shape.unsqueeze (transpose b) (-3))
  in
  Shape.rop p Op.Add [ Shape.ndim p - 1 ]

(* Householder QR

   Step [i] reflects the part of column [i] of [r] on and below the diagonal,
   [x], onto the diagonal: by [I - w vᵀ], [v] being 1 at row [i] and [x] scaled
   below it, applied to the rows of [r] from [i] on, to the right of column [i],
   and accumulated in [q] on the right. Column [i] itself takes its reflected
   value, the diagonal element and zeros below it. The diagonal takes the sign
   opposite to [x]'s first element, so that no cancellation occurs, and a column
   already zero below the diagonal takes no reflection. *)

let sum_last u = Shape.unsqueeze (Shape.rop u Op.Add [ Shape.ndim u - 1 ]) (-1)
let max_last u = Shape.unsqueeze (Shape.rop u Op.Max [ Shape.ndim u - 1 ]) (-1)

let householder device a =
  let batch, m, n = matrix a in
  let idx = Shape.arange ~dtype:Int32 m in
  let rows = row_index m and columns = column_index n in
  let reflect q r i =
    let at_i = Ops.eq idx i in
    let c = Shape.squeeze ~axis:(-1) (at (Shape.ndim r - 1) r i) in
    let x = Ops.where (Ops.ge idx i) c (zero c) in
    (* The reflector is built from the column divided by its largest magnitude,
       so that no square, sum or quotient overflows or underflows; only the
       diagonal element takes the column's scale back. *)
    let magnitude = abs x in
    let largest = max_last magnitude in
    let scale =
      Ops.where (Ops.ne largest (zero largest)) largest (float largest 1.)
    in
    let scaled = fdiv x scale in
    let norm = Ops.sqrt (sum_last (Ops.mul scaled scaled)) in
    let x0 = sum_last (Ops.where at_i x (zero x)) in
    let s0 = sum_last (Ops.where at_i scaled (zero scaled)) in
    let sgn = direction x0 in
    (* A column already zero below the diagonal takes no reflection, and keeps
       its diagonal element's sign. *)
    let below = sum_last (Ops.where (Ops.gt idx i) magnitude (zero x)) in
    let active = Ops.ne below (zero below) in
    let u0 = Ops.O.(s0 + (sgn * norm)) in
    let v =
      fdiv (Ops.where at_i u0 scaled) (Ops.where active u0 (float u0 1.))
    in
    let v = Shape.unsqueeze v (-1) in
    let tau =
      Ops.where active
        (fdiv (Ops.mul sgn u0) (Ops.where active norm (float norm 1.)))
        (zero norm)
    in
    let w = Ops.mul (Shape.unsqueeze tau (-1)) v in
    let diagonal =
      Ops.where active
        (Ops.mul (Ops.mul sgn (float sgn (-1.))) (Ops.mul scale norm))
        x0
    in
    let reflected =
      Ops.where (Ops.eq rows i)
        (Shape.unsqueeze diagonal (-1))
        (Ops.where (Ops.lt rows i) r (zero r))
    in
    let applied = Ops.sub r (dot w (Ops.contiguous (dot (transpose v) r))) in
    (* A column that takes no reflection leaves q and r as they are: its v holds
       what the column held, and a NaN or an infinity there would spread through
       a product with a zero tau. *)
    let on = Shape.unsqueeze active (-1) in
    [
      Ops.where on (Ops.sub q (dot (Ops.contiguous (dot q v)) (transpose w))) q;
      Ops.where (Ops.eq columns i) reflected
        (Ops.where (Ops.ge rows i) (Ops.where on applied r) r);
    ]
  in
  let step i = function
    | [ q; r ] -> reflect q r i
    | _ -> invalid_arg "Lower_linalg.householder"
  in
  let q = Shape.expand (eye (dtype a) m m) (ints (batch @ [ m; m ])) in
  match repeat device (Int.min m n) [ q; a ] step with
  | [ q; r ] -> (q, r)
  | _ -> invalid_arg "Lower_linalg.householder"

(* The upper triangle of the matrices of [u], [+0] below it. *)
let triu u =
  let _, m, n = matrix u in
  Ops.where (Ops.le (row_index m) (column_index n)) u (zero u)

let qr device ~complete a =
  let _, m, n = matrix a in
  let k = Int.min m n in
  let q, r = householder device a in
  let r = triu r in
  if complete then (q, r)
  else (block q None (Some (0, k)), block r (Some (0, k)) None)

(* Jacobi rotations

   A round rotates [n / 2] disjoint pairs of positions of an axis, each by the
   angle that zeroes the pair's off-diagonal element. The pairs are a
   round-robin tournament's by the circle method: position [k] meets position [n
   - 1 - k], the middle one of an odd [n] sitting the round out, and after the
   round each position but the first moves one place along the others, every
   position for an odd [n]. Values are kept in the order of the positions, so
   that a round reads and writes each element once; the sort of the result
   undoes it. For six positions, the first round meets [(0, 5)], [(1, 4)] and
   [(2, 3)], and the positions then hold [0 5 1 2 3 4], so that the second meets
   [(0, 4)], [(5, 3)] and [(1, 2)].

   A loop repeats the round, which the program holds at most three times
   whatever the count. A round's kernels compute its rotations, then rotate each
   value once. *)

(* The least [k] with [2^k >= n]. *)
let ceil_log2 n =
  let rec go k = if 1 lsl k >= n then k else go (k + 1) in
  go 0

(* The rounds of a sweep, which meets every pair of [n] positions once: [n - 1],
   or [n] for an odd [n], whose rounds each leave one out. *)
let sweep n = n - 1 + (n mod 2)

(* [tangents ~alpha ~beta ~gamma] is the tangent [t] of the rotation that zeroes
   the off-diagonal element of each symmetric [[alpha, gamma], [gamma, beta]],
   of an angle at most a quarter turn in magnitude: the root of [t² + 2 tau t -
   1] of least magnitude, [tau = (beta - alpha) / 2 gamma]. It is zero where
   [gamma] is, or where [tau²] overflows, as its limit [1 / 2 tau] is then below
   the roundoff. *)
let tangents ~alpha ~beta ~gamma =
  let rot = Ops.ne gamma (zero gamma) in
  let one = float gamma 1. and two = float gamma 2. in
  let tau = fdiv Ops.O.(beta - alpha) Ops.O.(two * Ops.where rot gamma one) in
  let t = fdiv (direction tau) Ops.O.(abs tau + Ops.sqrt (one + (tau * tau))) in
  Ops.where rot t (zero t)

(* [rotations n t] is the cosines and the sines, along the last axis, of a round
   of [n] positions whose pairs [(k, n - 1 - k)] turn by the tangents [t] at
   [k], computed into storage of their own: the cosines hold [c = 1 / sqrt (1 +
   t²)] at both positions of a pair, the sines [-c t] at [k] and [c t] at its
   mirror image, and the middle position of an odd [n] [1] and [0]. Each
   position reads its pair's tangent once. *)
let rotations n t =
  let r = Shape.ndim t and h = n / 2 in
  let k = Shape.arange ~dtype:Int32 n in
  let first = Ops.lt k (Ops.int h) and mirror = Ops.ge k (Ops.int (n - h)) in
  let along u =
    let shape = List.mapi (fun d s -> if d = r - 1 then Ops.Int n else s) in
    Shape.expand
      (Shape.reshape u
         (ints (List.init r (fun d -> if d = r - 1 then n else 1))))
      (shape (Shape.shape t))
  in
  let pair =
    Ops.where mirror
      (Ops.sub (Ops.int (n - 1)) k)
      (Ops.where first k (Ops.int 0))
  in
  let constant x = Ops.const ~dtype:(dtype t) (`Float x) in
  let sign =
    Ops.where first (constant (-1.))
      (Ops.where mirror (constant 1.) (constant 0.))
  in
  let t = Ops.mul (take t (r - 1) (along pair)) (along sign) in
  let one = float t 1. in
  let c = fdiv one (Ops.sqrt Ops.O.(one + (t * t))) in
  let both =
    Ops.contiguous
      (Shape.cat ~axis:(r - 1)
         (Shape.unsqueeze c (r - 1))
         [ Shape.unsqueeze (Ops.mul c t) (r - 1) ])
  in
  let row k = Shape.squeeze ~axis:(r - 1) (part both (r - 1) k (k + 1)) in
  (row 0, row 1)

(* [turn u axis (cosines, sines)] rotates each pair of positions [k] and [n - 1
   - k] of [axis] of [u], each the other's mirror image, to [c u_k - s u_j] and
   [s u_k + c u_j]: [cosines] and [sines] multiply [u] and its mirror image. *)
let turn u axis (cosines, sines) =
  Ops.O.((cosines * u) + (sines * Shape.flip u [ axis ]))

(* [advance u axis] moves each position of [axis] but the first one place along
   the others, the last to the second: every position, the last to the first,
   for an odd size. Position [k] reads the one it moves from, a read that each
   element makes once. *)
let advance u axis =
  let n = List.nth (Shape.max_shape u) axis in
  let k = Shape.arange ~dtype:Int32 n in
  let from =
    if n mod 2 = 1 then Ops.O.((k + int (Int.pred n)) % int n)
    else
      let others = n - 1 and back = n - 3 in
      Ops.where (Ops.eq k (Ops.int 0)) k
        Ops.O.(((k + int back) % int others) + int 1)
  in
  let along = List.init (Shape.ndim u) (fun d -> if d = axis then n else 1) in
  take u axis (Shape.expand (Shape.reshape from (ints along)) (Shape.shape u))

(* [rotate u axis rotations] is a round's rotation of [axis] of [u], whose
   positions then advance. *)
let rotate u axis rotations = advance (turn u axis rotations) axis

(* The sweeps that bring a matrix of [n] rows to diagonal, or [n] columns to
   orthogonality, for a precision of [p] bits: [⌈p (⌈log2 n⌉ + extra) / 10⌉].
   Serial orderings of the pairs converge quadratically near the end, with
   repeated values too (Hari, as Drmač and Veselić's LAPACK Working Note 169
   recounts in its §1). This ordering gains only a roughly constant factor a
   sweep on repeated and graded spectra, 2 to 8 in a [float64] model of [svd]'s
   rounds up to [n = 128].

   Two-sided rotations take [extra = 0]: over repeated, clustered, graded and
   random spectra up to [n = 64] in [float32] and [float64], [0.83] of the
   count brought them to an off-diagonal norm of [0.43 n u], [u] the unit
   roundoff, against the [16 n u] a matrix fails beyond.

   One-sided rotations take [extra = 1]. Their convergence test, relative to
   each pair of columns, needed up to [0.9] of [extra = 0]'s count: 9 of 10
   sweeps in [float32] at [n = 16], on a spectrum graded down to twice the
   norm the test exempts. It needs at most [0.75] of [extra = 1]'s, from
   [n = 4] to [128] in [float32] and [float64], over graded, clustered and
   random spectra, spectra of half zeros and spectra of two or three values
   each repeated, each in 16 random bases up to [n = 20], 4 up to [64] and one
   at [128]. *)
let sweeps n p extra =
  if n < 2 then 0 else ((p * (ceil_log2 n + extra)) + 9) / 10

let rounds ?(extra = 0) n dt =
  let _, mantissa = Dtype.finfo dt in
  sweeps n (mantissa + 1) extra * sweep n

(* Singular values by one-sided Jacobi rotations

   The columns of [u], [r]'s leading square, are made orthogonal by rotating
   pairs of them, each by the angle that zeroes their inner product; [v]
   accumulates the rotations. [u] and [v] are square and are kept stacked, [u]
   above [v], so that a round rotates the columns of both at once.

   A pair whose inner product is within the unit roundoff of the product of its
   columns' norms takes no rotation: that is the product's rounding error, as a
   rotated column's error is relative to its own norm. Between two columns of
   one repeated singular value such a product is rounding error, as the
   difference of their norms is, so the angle they set is arbitrary, and the
   turn mixes the pair's products with other columns: those the round's other
   rotations zeroed with those they had not. A [float64] 16 x 16 matrix of
   singular values [1] and [0.9], each repeated, took 13 sweeps to bring its
   values within [16 n u] of an eager [svd]'s without it, and 11 with it.

   The interface's [g] sums three terms. The QRs of [a] and of the rotated
   columns give [q]s orthogonal to [k (2p + 10) u] an element. The rounds move
   the columns by at most [8 R u] of their norm, as each rotation of two columns
   moves them by at most [8 u] of it. The columns the convergence test exempts
   have norms up to [8 R (1 + √k) u ‖a‖_F]. As [g = γ(c)], the bound is vacuous
   once [c u >= 1/2], which the sizes the interface names reach. *)

(* [converged ~tol ~noise u] is whether each pair of columns of the matrices of
   [u] of norms [s_i] and [s_j] above [noise] times the matrix's norm has an
   inner product within [tol s_i s_j]: orthogonal to the roundoff of the inner
   product, relative to the columns themselves. A column at [noise] or below
   holds the roundoff the rotations left, not a direction; any direction serves
   it. [u] is divided by its largest magnitude first, so that no product
   overflows. *)
let converged ~tol ~noise u =
  let _, _, k = matrix u in
  let largest = over_matrix Op.Max (abs u) in
  let u =
    fdiv u
      (Ops.where (Ops.gt largest (zero largest)) largest (float largest 1.))
  in
  let g = dot (transpose u) u in
  let s = Ops.sqrt (diagonal g) in
  let norm = Ops.sqrt (Shape.rop (Ops.mul s s) Op.Add [ Shape.ndim s - 1 ]) in
  let big =
    Ops.lt (Shape.unsqueeze (Ops.mul norm (float norm noise)) (-1)) s
  in
  let both =
    Ops.bitwise_and (Shape.unsqueeze big (-1)) (Shape.unsqueeze big (-2))
  in
  let products = Ops.mul (Shape.unsqueeze s (-1)) (Shape.unsqueeze s (-2)) in
  let within =
    Ops.bitwise_or
      (Ops.bitwise_not both)
      (Ops.bitwise_or
         (Ops.eq (row_index k) (column_index k))
         (Ops.bitwise_not (Ops.gt (abs g) (Ops.mul products (float g tol)))))
  in
  let fails = Ops.cast (Ops.bitwise_not within) Int32 in
  Ops.eq (over_matrix Op.Add fails) (Ops.int 0)

let svd device ~complete a =
  let batch, m, n = matrix a in
  let num = Int.min m n and q_num = Int.max m n in
  let dt = dtype a in
  let q, r = householder device (if m >= n then a else transpose a) in
  let square k = Shape.expand (eye dt k k) (ints (batch @ [ k; k ])) in
  let rank = Shape.ndim r in
  let columns x = Shape.unsqueeze x (-2) in
  let round _ = function
    | [ y ] ->
        (* The columns of the pairs, each one's mate alongside it: one kernel
           sums their products and computes the tangents from the sums. *)
        let u = part y (rank - 2) 0 num in
        let first x = part x (rank - 1) 0 (num / 2) in
        let a = first u and b = first (Shape.flip u [ rank - 1 ]) in
        let sums x = Shape.rop x Op.Add [ rank - 2 ] in
        let alpha = sums (Ops.mul a a) and beta = sums (Ops.mul b b) in
        let gamma = sums (Ops.mul a b) in
        let floor =
          Ops.mul
            (Ops.mul (Ops.sqrt alpha) (Ops.sqrt beta))
            (float alpha (roundoff dt))
        in
        let above = Ops.gt (abs gamma) floor in
        let cosines, sines =
          rotations num
            (Ops.contiguous
               (tangents ~alpha ~beta
                  ~gamma:(Ops.where above gamma (zero gamma))))
        in
        [ rotate y (rank - 1) (columns cosines, columns sines) ]
    | _ -> invalid_arg "Lower_linalg.svd"
  in
  (* The rotations square r's columns: r is divided by the power of two of its
     largest magnitude, exactly, so that no square overflows or underflows,
     and the singular values take it back. *)
  let r = block (triu r) (Some (0, num)) (Some (0, num)) in
  let e = exponent r in
  let r = Lower_real.scale r (Shape.expand (negate e) (Shape.shape r)) in
  let rounds = rounds ~extra:1 num dt in
  let y =
    List.hd
      (repeat device rounds
         [ Shape.cat ~axis:(rank - 2) r [ square num ] ]
         round)
  in
  let u = part y (rank - 2) 0 num in
  (* A matrix fails if it holds NaN or an infinity, or if its rotated columns
     are not orthogonal: once they are, each inner product is within the
     roundoff of a sum of [num] terms, relative to its columns. A column of the
     rotations' roundoff is exempt. *)
  let k = float_of_int num in
  let noise =
    8. *. float_of_int rounds *. (1. +. Float.sqrt k) *. roundoff dt
  in
  let ok =
    Ops.bitwise_and (finite a)
      (converged ~tol:(16. *. k *. roundoff dt) ~noise u)
  in
  let norms = Ops.sqrt (Shape.rop (Ops.mul u u) Op.Add [ rank - 2 ]) in
  let order = order ~descending:true norms in
  let s = take norms (rank - 2) order in
  let s =
    defined ok
      (Lower_real.scale s
         (Shape.expand (Shape.squeeze ~axis:(-1) e) (Shape.shape s)))
  in
  let v = by_order order (part y (rank - 2) num (2 * num)) in
  (* The sorted columns are orthogonal: the reflections that triangularize them
     are an orthogonal basis of which each column, with the sign it takes on the
     diagonal, is the direction, and which completes the columns of zero
     norm. *)
  let q_u, r_u = householder device (by_order order u) in
  let u = Ops.mul q_u (Shape.unsqueeze (direction (diagonal r_u)) (-2)) in
  let inside =
    Ops.bitwise_and
      (Ops.lt (row_index q_num) (Ops.int num))
      (Ops.lt (column_index q_num) (Ops.int num))
  in
  let pad =
    List.map (fun _ -> None) batch
    @ [
        Some (Ops.Int 0, Ops.Int (q_num - num));
        Some (Ops.Int 0, Ops.Int (q_num - num));
      ]
  in
  let u = dot q (Ops.where inside (Shape.pad u pad) (square q_num)) in
  let u = if complete then u else block u None (Some (0, num)) in
  let u, vh = if m >= n then (u, transpose v) else (v, transpose u) in
  [| s; defined ok u; defined ok vh |]

(* Hermitian eigenvalues by two-sided Jacobi rotations

   A round rotates pairs of rows and the same columns of the symmetric [x], each
   by the angle that zeroes the pair's off-diagonal element, and the same
   columns of [v], which accumulates the rotations.

   An element within the unit roundoff of [x]'s norm takes no rotation. Between
   two diagonal elements of one repeated eigenvalue such an element is rounding
   error, as their difference is, so the angle they set is arbitrary, and the
   turn mixes the pair's couplings to other eigenvalues: those the round's other
   rotations zeroed with those they had not. Convergence then falls to a bit a
   sweep: a [float32] 5 x 5 matrix of eigenvalues -1, 1 and 2 took 13 sweeps
   to reach [4 n u], against its 8. The element left is within the roundoff
   the sweeps reach. *)

let eigh device ~vectors a =
  let dt = dtype a in
  let batch, n, _ = matrix a in
  let r = Shape.ndim a in
  (* The lower triangle, mirrored: the upper one is never read. The rotations
     run on it divided by the power of two of its largest magnitude, and the
     eigenvalues take it back: no difference of two diagonal elements, rotated
     element or square then overflows. The division rounds only an element
     whose quotient is subnormal, and only a square far below the floor
     underflows. *)
  let lower = Ops.ge (row_index n) (column_index n) in
  let x = Ops.where lower a (transpose a) in
  let e = exponent x in
  let x = Lower_real.scale x (Shape.expand (negate e) (Shape.shape x)) in
  let floor =
    Shape.squeeze ~axis:(-1) (norm_roundoff (Ops.where lower x (zero x)))
  in
  let rows u = Shape.unsqueeze u (-1) and columns u = Shape.unsqueeze u (-2) in
  let round _ = function
    | x :: v ->
        let d = diagonal x and first u = part u (r - 2) 0 (n / 2) in
        let gamma = first (diagonal (Shape.flip x [ r - 1 ])) in
        let above =
          Ops.gt (abs gamma) (Shape.expand floor (Shape.shape gamma))
        in
        let cosines, sines =
          rotations n
            (tangents ~alpha:(first d)
               ~beta:(first (Shape.flip d [ r - 2 ]))
               ~gamma:(Ops.where above gamma (zero gamma)))
        in
        let along axis per u = rotate u axis (per cosines, per sines) in
        along (r - 1) columns (along (r - 2) rows x)
        :: List.map (along (r - 1) columns) v
    | [] -> []
  in
  let v =
    if vectors then [ Shape.expand (eye dt n n) (ints (batch @ [ n; n ])) ]
    else []
  in
  let rotated = repeat device (rounds n dt) (x :: v) round in
  (* A matrix fails if its read triangle holds NaN or an infinity, or if the
     sweeps leave it off the diagonal beyond the [n] units of roundoff of its
     norm they reach. *)
  let ok =
    Ops.bitwise_and (finite x)
      (diagonal_within (16. *. float_of_int n *. roundoff dt) (List.hd rotated))
  in
  let w = diagonal (List.hd rotated) in
  let order = order ~descending:false w in
  let w = take w (r - 2) order in
  let w =
    defined ok
      (Lower_real.scale w
         (Shape.expand (Shape.squeeze ~axis:(-1) e) (Shape.shape w)))
  in
  match List.tl rotated with
  | [] -> [| w |]
  | v :: _ -> [| w; defined ok (by_order order v) |]

(* LU with partial pivoting

   Step [j] takes as pivot the first row of largest magnitude in column [j] on
   or below the diagonal, a NaN on the diagonal being the largest and one below
   it the least, and swaps it with row [j], moving each element with its bits.
   The column below the pivot is divided by it, unless it is zero, and the
   trailing rows take the rank-one update. Step [j] writes its pivot's row at
   position [j] of the pivots. *)

let lu device a =
  let batch, m, n = matrix a in
  let k = Int.min m n in
  let rank = Shape.ndim a in
  let index = row_index m and cols = column_index n in
  let positions = Shape.arange ~dtype:Int32 k in
  (* [swap u j p] exchanges row [j] and row [p], one per matrix, of [u]. *)
  let swap u j p =
    let _, _, c = matrix u in
    let row_p = take u (rank - 2) (Shape.expand p (ints (batch @ [ 1; c ]))) in
    Ops.where (Ops.eq index j) row_p
      (Ops.where (Ops.eq index p) (at (rank - 2) u j) u)
  in
  let step j = function
    | [ x; perm; pivots ] ->
        let col = at (rank - 1) x j in
        let never = float col Float.neg_infinity in
        let magnitude =
          Ops.where (Ops.lt index j) never
            (Ops.where (Ops.ne col col)
               (Ops.where (Ops.eq index j) (float col Float.infinity) never)
               (abs col))
        in
        let p = first_max (rank - 2) magnitude in
        let x = swap x j p and perm = swap perm j p in
        let col = at (rank - 1) x j in
        let pivot = at (rank - 2) col j in
        let l = Ops.where (Ops.ne pivot (zero pivot)) (fdiv col pivot) col in
        let below = Ops.gt index j in
        let x =
          Ops.where
            (Ops.bitwise_and below (Ops.eq cols j))
            l
            (Ops.where
               (Ops.bitwise_and below (Ops.gt cols j))
               (Ops.sub x (Ops.mul l (at (rank - 2) x j)))
               x)
        in
        let p = Ops.cast (Shape.squeeze ~axis:(-1) p) Int64 in
        let pivots =
          Ops.where (Ops.eq positions j)
            (Shape.expand p (Shape.shape pivots))
            pivots
        in
        [ x; perm; pivots ]
    | _ -> invalid_arg "Lower_linalg.lu"
  in
  let perm = Shape.expand (Ops.cast index Int64) (ints (batch @ [ m; 1 ])) in
  let pivots =
    Shape.expand
      (Ops.const ~dtype:Int64 (`Int Bigint.zero))
      (ints (batch @ [ k ]))
  in
  match repeat device k [ a; perm; pivots ] step with
  | [ x; perm; pivots ] -> [| x; pivots; Shape.squeeze ~axis:(-1) perm |]
  | _ -> invalid_arg "Lower_linalg.lu"

(* Cholesky

   Step [j] takes column [j] of the working matrix, on and below the diagonal,
   as column [j] of [L]: its diagonal element's square root heads it, and the
   rest is divided by that root. The trailing rows and columns then lose the
   product of the column with itself. Only the lower triangle is ever read. A
   pivot [d] is positive exactly where its root is: [sqrt] of a negative or NaN
   [d] is NaN, of [+0] or [-0] a zero. *)

let cholesky device ~upper a =
  let _, n, _ = matrix a in
  let rank = Shape.ndim a in
  let rows = row_index n and cols = column_index n in
  let lower = Ops.ge rows cols in
  let step j = function
    | [ s ] ->
        let c = at (rank - 1) s j in
        let root = Ops.sqrt (at (rank - 2) c j) in
        let l =
          Ops.where (Ops.eq rows j)
            (Shape.expand root (Shape.shape c))
            (Ops.where (Ops.gt rows j) (fdiv c root) (zero root))
        in
        let trailing = Ops.bitwise_and (Ops.gt rows j) (Ops.gt cols j) in
        [
          Ops.where (Ops.eq cols j) l
            (Ops.where trailing (Ops.sub s (Ops.mul l (transpose l))) s);
        ]
    | _ -> invalid_arg "Lower_linalg.cholesky"
  in
  let s = List.hd (repeat device n [ Ops.where lower a (zero a) ] step) in
  let l = Ops.where lower s (zero s) in
  let d = diagonal l in
  let failed = Ops.cast (Ops.logical_not (Ops.gt d (zero d))) Int32 in
  let none = Ops.eq (Shape.rop failed Op.Add [ rank - 2 ]) (Ops.int 0) in
  let ok = Shape.unsqueeze (Shape.unsqueeze none (-1)) (-1) in
  defined ok (if upper then transpose l else l)

(* Triangular solve

   The system is made lower triangular, [aᵀ] under [adjoint], and reversed
   along both axes, with [b]'s rows, when the triangle it reads is the upper
   one. Step [i] divides row [i] of the right-hand sides by the diagonal
   element, which makes it row [i] of the solution, and takes its product with
   column [i] of the matrix from the rows below. A zero diagonal element makes
   the matrix singular. *)

let solve_triangular device ~upper ~adjoint ~unit_diagonal a b =
  let m = if adjoint then transpose a else a in
  let rank = Shape.ndim m in
  let reversed = upper <> adjoint in
  let m = if reversed then Shape.flip m [ rank - 2; rank - 1 ] else m in
  let b = if reversed then Shape.flip b [ rank - 2 ] else b in
  let batch, n, _ = matrix m in
  let rows = row_index n in
  let step i = function
    | [ x ] ->
        let c = at (rank - 1) m i in
        let xi = at (rank - 2) x i in
        let xi = if unit_diagonal then xi else fdiv xi (at (rank - 2) c i) in
        [
          Ops.where (Ops.eq rows i)
            (Shape.expand xi (Shape.shape x))
            (Ops.where (Ops.gt rows i) (Ops.sub x (Ops.mul c xi)) x);
        ]
    | _ -> invalid_arg "Lower_linalg.solve_triangular"
  in
  let x = List.hd (repeat device n [ b ] step) in
  let x = if reversed then Shape.flip x [ rank - 2 ] else x in
  if unit_diagonal || n = 0 then x
  else
    let d = diagonal m in
    let zeros = Ops.cast (Ops.eq d (zero d)) Int32 in
    let none = Ops.eq (Shape.rop zeros Op.Add [ rank - 2 ]) (Ops.int 0) in
    defined (Shape.reshape none (ints (batch @ [ 1; 1 ]))) x

(* Operations *)

let real (type v s) name (dt : (v, s) D.t) =
  if D.is D.Complex dt then unlowered ("a complex " ^ name)

let op ~device:d (node : 'v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) (type d r)
    (l : (d, r) P.linalg) : Ops.t array =
  let operand name a =
    real name (Nx.dtype a);
    node a
  in
  match l with
  | Cholesky { triangle; a } ->
      [| cholesky d ~upper:(triangle = Upper) (operand "Cholesky" a) |]
  | Lu a -> lu d (operand "Lu" a)
  | Qr { factors; a } ->
      let q, r = qr d ~complete:(factors = Complete) (operand "Qr" a) in
      [| q; r |]
  | Svd_values { a; _ } ->
      [| (svd d ~complete:false (operand "Svd_values" a)).(0) |]
  | Svd { factors; a; _ } ->
      svd d ~complete:(factors = Complete) (operand "Svd" a)
  | Eigh_values { a; _ } -> eigh d ~vectors:false (operand "Eigh_values" a)
  | Eigh { a; _ } -> eigh d ~vectors:true (operand "Eigh" a)
  | Solve_triangular { triangle; adjoint; unit_diagonal; a; b } ->
      let a = operand "Solve_triangular" a in
      let b = operand "Solve_triangular" b in
      [|
        solve_triangular d ~upper:(triangle = Upper) ~adjoint ~unit_diagonal a
          b;
      |]
  | Eig_values _ -> unlowered "Eig_values"
  | Eig _ -> unlowered "Eig"
