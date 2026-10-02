(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk

let dtype = Ops.dtype
let ints l = List.map (fun n -> Ops.Int n) l
let zero u = Ops.const_like u (`Float 0.)
let float u x = Ops.const_like u (`Float x)
let transpose u = Ops.transpose u (-2) (-1)

(* The batch axes of the matrices of [u], their rows and their columns. *)
let matrix u =
  let dims = Ops.max_shape u in
  let r = List.length dims in
  ( List.filteri (fun i _ -> i < r - 2) dims,
    List.nth dims (r - 2),
    List.nth dims (r - 1) )

(* [block u rows cols] is the rows and columns of the matrices of [u] in the
   bounds [rows] and [cols], each all of them if [None]. *)
let block u rows cols =
  let r = Ops.ndim u in
  let bound = Option.map (fun (lo, hi) -> (Ops.Int lo, Ops.Int hi)) in
  Ops.shrink u
    (List.init r (fun i ->
         if i = r - 2 then bound rows
         else if i = r - 1 then bound cols
         else None))

(* The diagonal of the matrices of [u]: the first column of their elements laid
   out in rows of one more element. *)
let diagonal u =
  let batch, n, _ = matrix u in
  let flat = Ops.reshape u (ints (batch @ [ n * n ])) in
  let padded =
    Ops.pad flat
      (List.map (fun _ -> None) batch @ [ Some (Ops.Int 0, Ops.Int n) ])
  in
  Ops.reshape
    (block
       (Ops.reshape padded (ints (batch @ [ n; n + 1 ])))
       None
       (Some (0, 1)))
    (ints (batch @ [ n ]))

let row u i = block u (Some (i, i + 1)) None
let column u j = block u None (Some (j, j + 1))
let entry u i j = block u (Some (i, i + 1)) (Some (j, j + 1))

(* The row and the column of each element of a matrix of [m] rows and [n]
   columns. *)
let row_index m = Ops.reshape (Ops.arange ~dtype:Int32 m) (ints [ m; 1 ])
let column_index n = Ops.reshape (Ops.arange ~dtype:Int32 n) (ints [ 1; n ])
let is u i = Ops.eq u (Ops.int i)
let eye dt n m = Ops.cast (Ops.eq (row_index n) (Ops.arange ~dtype:Int32 m)) dt

(* -1 below zero and 1 elsewhere: the sign of a nonzero element, and 1 for a
   zero one. *)
let direction x = Ops.where (Ops.lt x (zero x)) (float x (-1.)) (float x 1.)
let fdiv x y = Lower_arith.binary Fdiv x (Ops.expand y (Ops.shape x))

(* Products

   [dot a b] multiplies the matrices of [a] and [b] as the reference's [dot]
   does: each row of [a] against each column of [b] along a new last axis,
   broadcast over the batch axes, multiplied and summed by [sum]. *)

let dot ?(sum = fun u axis -> Ops.rop u Op.Add [ axis ]) a b =
  let p = Ops.mul (Ops.unsqueeze a (-2)) (Ops.unsqueeze (transpose b) (-3)) in
  sum p (Ops.ndim p - 1)

let matmul a b =
  let dt = dtype a in
  let acc = Lower_reduce.accumulator dt in
  let sum u axis = Lower_reduce.reduce Sum ~axes:[ axis ] u in
  Ops.cast (dot ~sum (Ops.cast a acc) (Ops.cast b acc)) dt

(* Householder QR

   Step [i] reflects the part of column [i] of [r] on and below the diagonal,
   [x], onto the diagonal: by [I - w vᵀ], [v] being 1 at row [i] and [x] scaled
   below it, applied to the rows of [r] from [i] on, to the right of column [i],
   and accumulated in [q] on the right. Column [i] itself takes its reflected
   value, the diagonal element and zeros below it. The diagonal takes the sign
   opposite to [x]'s first element, so that no cancellation occurs, and a column
   already zero below the diagonal takes no reflection. A loop repeats the step
   ({!Loop.repeat}), which reads [i] from a count it carries. *)

let sum_last u = Ops.unsqueeze (Ops.rop u Op.Add [ Ops.ndim u - 1 ]) (-1)
let max_last u = Ops.unsqueeze (Ops.rop u Op.Max [ Ops.ndim u - 1 ]) (-1)

let householder ~device a =
  let batch, m, n = matrix a in
  let idx = Ops.arange ~dtype:Int32 m in
  let rows = row_index m and columns = column_index n in
  let reflect q r i =
    let at_i = Ops.eq idx i in
    let c =
      Ops.squeeze ~axis:(-1)
        (Lower_index.gather
           (Ops.ndim r - 1)
           (Ops.expand i (ints (batch @ [ m; 1 ])))
           r)
    in
    let x = Ops.where (Ops.ge idx i) c (zero c) in
    (* The reflector is built from the column divided by its largest magnitude,
       so that no square, sum or quotient overflows or underflows; only the
       diagonal element takes the column's scale back. *)
    let magnitude =
      Ops.where (Ops.lt x (zero x)) (Ops.mul x (float x (-1.))) x
    in
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
    let v = Ops.unsqueeze v (-1) in
    let tau =
      Ops.where active
        (fdiv (Ops.mul sgn u0) (Ops.where active norm (float norm 1.)))
        (zero norm)
    in
    let w = Ops.mul (Ops.unsqueeze tau (-1)) v in
    let diagonal =
      Ops.where active
        (Ops.mul (Ops.mul sgn (float sgn (-1.))) (Ops.mul scale norm))
        x0
    in
    let reflected =
      Ops.where (Ops.eq rows i)
        (Ops.unsqueeze diagonal (-1))
        (Ops.where (Ops.lt rows i) r (zero r))
    in
    let applied = Ops.sub r (dot w (Ops.contiguous (dot (transpose v) r))) in
    (* A column that takes no reflection leaves q and r as they are: its v holds
       what the column held, and a NaN or an infinity there would spread through
       a product with a zero tau. *)
    let on = Ops.unsqueeze active (-1) in
    [
      Ops.where on (Ops.sub q (dot (Ops.contiguous (dot q v)) (transpose w))) q;
      Ops.where (Ops.eq columns i) reflected
        (Ops.where (Ops.ge rows i) (Ops.where on applied r) r);
      Ops.add i (Ops.const_like i (`Int Bigint.one));
    ]
  in
  let step = function
    | [ q; r; i ] -> reflect q r i
    | _ -> invalid_arg "Lower_linalg.householder"
  in
  let q = Ops.expand (eye (dtype a) m m) (ints (batch @ [ m; m ])) in
  match
    Loop.repeat device (Int.min m n)
      [ q; a; Ops.const ~dtype:Int32 (`Int Bigint.zero) ]
      step
  with
  | q :: r :: _ -> (q, r)
  | _ -> invalid_arg "Lower_linalg.householder"

(* The upper triangle of the matrices of [u], zeros below it. *)
let triu u =
  let _, m, n = matrix u in
  Ops.where (Ops.le (row_index m) (column_index n)) u (zero u)

let qr ~device ~reduced a =
  let dt = dtype a in
  let _, m, n = matrix a in
  let k = Int.min m n in
  let q, r = householder ~device (Lower_arith.widen a) in
  let r = triu r in
  let q, r =
    if reduced then (block q None (Some (0, k)), block r (Some (0, k)) None)
    else (q, r)
  in
  (Ops.cast q dt, Ops.cast r dt)

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

   A program holds one round, which a loop repeats ({!Loop.repeat}): compiling
   costs a round whatever the count. A round's kernels compute its rotations,
   then rotate each value once. *)

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
  let t =
    fdiv (direction tau)
      Ops.O.(Lower_arith.unary Abs tau + Ops.sqrt (one + (tau * tau)))
  in
  Ops.where rot t (zero t)

(* [part u axis lo hi] is the elements [lo] to [hi], excluded, of [axis]. *)
let part u axis lo hi =
  Ops.shrink u
    (List.init (Ops.ndim u) (fun d ->
         if d = axis then Some (Ops.Int lo, Ops.Int hi) else None))

(* [rotations n t] is the cosines and the sines, along the last axis, of a round
   of [n] positions whose pairs [(k, n - 1 - k)] turn by the tangents [t] at
   [k], computed into storage of their own: the cosines hold [c = 1 / sqrt (1 +
   t²)] at both positions of a pair, the sines [-c t] at [k] and [c t] at its
   mirror image, and the middle position of an odd [n] [1] and [0]. Each
   position reads its pair's tangent once. *)
let rotations n t =
  let r = Ops.ndim t and h = n / 2 in
  let k = Ops.arange ~dtype:Int32 n in
  let first = Ops.lt k (Ops.int h) and mirror = Ops.ge k (Ops.int (n - h)) in
  let along u =
    let shape = List.mapi (fun d s -> if d = r - 1 then Ops.Int n else s) in
    Ops.expand
      (Ops.reshape u (ints (List.init r (fun d -> if d = r - 1 then n else 1))))
      (shape (Ops.shape t))
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
  let t = Ops.mul (Lower_index.gather (r - 1) (along pair) t) (along sign) in
  let one = float t 1. in
  let c = fdiv one (Ops.sqrt Ops.O.(one + (t * t))) in
  let both =
    Ops.contiguous
      (Ops.cat ~axis:(r - 1)
         (Ops.unsqueeze c (r - 1))
         [ Ops.unsqueeze (Ops.mul c t) (r - 1) ])
  in
  let row k = Ops.squeeze ~axis:(r - 1) (part both (r - 1) k (k + 1)) in
  (row 0, row 1)

(* [turn u axis (cosines, sines)] rotates each pair of positions [k] and [n - 1
   - k] of [axis] of [u], each the other's mirror image, to [c u_k - s u_j] and
   [s u_k + c u_j]: [cosines] and [sines] multiply [u] and its mirror image. *)
let turn u axis (cosines, sines) =
  Ops.O.((cosines * u) + (sines * Ops.flip u [ axis ]))

(* [advance u axis] moves each position of [axis] but the first one place along
   the others, the last to the second: every position, the last to the first,
   for an odd size. Position [k] reads the one it moves from, a gather that each
   element reads once. *)
let advance u axis =
  let n = List.nth (Ops.max_shape u) axis in
  let k = Ops.arange ~dtype:Int32 n in
  let from =
    if n mod 2 = 1 then Ops.O.((k + int (Int.pred n)) % int n)
    else
      let others = n - 1 and back = n - 3 in
      Ops.where (is k 0) k Ops.O.(((k + int back) % int others) + int 1)
  in
  let along = List.init (Ops.ndim u) (fun d -> if d = axis then n else 1) in
  Lower_index.gather axis
    (Ops.expand (Ops.reshape from (ints along)) (Ops.shape u))
    u

(* [rotate u axis (cosines, sines)] is a round's rotation of [axis] of [u],
   whose positions then advance. *)
let rotate u axis rotations = advance (turn u axis rotations) axis

(* Singular values by one-sided Jacobi rotations

   The columns of [u], [r]'s leading square, are made orthogonal by rotating
   pairs of them, each by the angle that zeroes their inner product; [v]
   accumulates the rotations. [u] and [v] are square and are kept stacked, [u]
   above [v], so that a round rotates the columns of both at once. *)

(* The rounds that bring [num] columns to orthogonality. The rotations converge
   quadratically once the columns are nearly orthogonal, and the sweeps it takes
   to get there grow as the logarithm of [num]. *)
let rounds num = if num < 2 then 0 else (ceil_log2 num + 3) * sweep num

let svd ~device ~full_matrices a =
  let dt = dtype a in
  let batch, m, n = matrix a in
  let num = Int.min m n and q_num = Int.max m n in
  let x = Lower_arith.widen a in
  let wdt = dtype x in
  let q, r = householder ~device (if m >= n then x else transpose x) in
  let square k = Ops.expand (eye wdt k k) (ints (batch @ [ k; k ])) in
  let rank = Ops.ndim r in
  let columns x = Ops.unsqueeze x (-2) in
  let round = function
    | [ y ] ->
        (* The columns of the pairs, each one's mate alongside it: one kernel
           sums their products and computes the tangents from the sums. *)
        let u = part y (rank - 2) 0 num in
        let first x = part x (rank - 1) 0 (num / 2) in
        let a = first u and b = first (Ops.flip u [ rank - 1 ]) in
        let sums x = Ops.rop x Op.Add [ rank - 2 ] in
        let cosines, sines =
          rotations num
            (Ops.contiguous
               (tangents
                  ~alpha:(sums (Ops.mul a a))
                  ~beta:(sums (Ops.mul b b))
                  ~gamma:(sums (Ops.mul a b))))
        in
        [ rotate y (rank - 1) (columns cosines, columns sines) ]
    | _ -> invalid_arg "Lower_linalg.svd"
  in
  let y =
    List.hd
      (Loop.repeat device (rounds num)
         [
           Ops.cat ~axis:(rank - 2)
             (block (triu r) (Some (0, num)) (Some (0, num)))
             [ square num ];
         ]
         round)
  in
  let u = part y (rank - 2) 0 num and v = part y (rank - 2) num (2 * num) in
  let norms = Ops.sqrt (Ops.rop (Ops.mul u u) Op.Add [ rank - 2 ]) in
  let order = Lower_reduce.argsort ~descending:true ~axis:(rank - 2) norms in
  let s = Lower_index.gather (rank - 2) order norms in
  let by_order x =
    Lower_index.gather (rank - 1)
      (Ops.expand (Ops.unsqueeze order (-2)) (ints (batch @ [ num; num ])))
      x
  in
  (* The sorted columns are orthogonal: the reflections that triangularize them
     are an orthogonal basis of which each column, with the sign it takes on the
     diagonal, is the direction, and which completes the columns of zero
     norm. *)
  let q_u, r_u = householder ~device (by_order u) in
  let u = Ops.mul q_u (Ops.unsqueeze (direction (diagonal r_u)) (-2)) in
  let v = by_order v in
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
  let u = dot q (Ops.where inside (Ops.pad u pad) (square q_num)) in
  let u = if full_matrices then u else block u None (Some (0, num)) in
  let u, vt = if m >= n then (u, transpose v) else (v, transpose u) in
  (Ops.cast u dt, Ops.cast s Float64, Ops.cast vt dt)

(* Symmetric eigenvalues by two-sided Jacobi rotations

   A round rotates pairs of rows and the same columns of the symmetric [x], each
   by the angle that zeroes the pair's off-diagonal element, and the same
   columns of [v], which accumulates the rotations. *)

(* The sweeps that bring a matrix of [n] rows to diagonal, for a precision of
   [p] bits. Two-sided Jacobi converges quadratically on distinct eigenvalues,
   but only linearly on repeated ones, a constant number of bits a sweep, fewer
   as [n] grows: over repeated, clustered, graded and ill-conditioned spectra,
   reaching an off-diagonal norm of [4 n u] took at most [0.83] of [⌈p ⌈log2 n⌉
   / 10⌉] sweeps, [u] the unit roundoff, up to [n = 64] in [float32] and
   [float64]. *)
let sweeps n p = if n < 2 then 0 else ((p * ceil_log2 n) + 9) / 10

let eigh ~device ~vectors a =
  let x = Lower_arith.widen a in
  let wdt = dtype x in
  let batch, n, _ = matrix x in
  let r = Ops.ndim x in
  (* The lower triangle, mirrored: the upper one is never read. *)
  let x = Ops.where (Ops.ge (row_index n) (column_index n)) x (transpose x) in
  let rows u = Ops.unsqueeze u (-1) and columns u = Ops.unsqueeze u (-2) in
  let round = function
    | x :: v ->
        let d = diagonal x and first u = part u (r - 2) 0 (n / 2) in
        let cosines, sines =
          rotations n
            (tangents ~alpha:(first d)
               ~beta:(first (Ops.flip d [ r - 2 ]))
               ~gamma:(first (diagonal (Ops.flip x [ r - 1 ]))))
        in
        let along axis per u = rotate u axis (per cosines, per sines) in
        along (r - 1) columns (along (r - 2) rows x)
        :: List.map (along (r - 1) columns) v
    | [] -> []
  in
  let _, mantissa = Dtype.finfo wdt in
  let v =
    if vectors then [ Ops.expand (eye wdt n n) (ints (batch @ [ n; n ])) ]
    else []
  in
  let rotated =
    Loop.repeat device (sweeps n (mantissa + 1) * sweep n) (x :: v) round
  in
  let w = diagonal (List.hd rotated) in
  let order = Lower_reduce.argsort ~descending:false ~axis:(r - 2) w in
  let w = Lower_index.gather (r - 2) order w in
  let v =
    List.map
      (fun v ->
        Ops.cast
          (Lower_index.gather (r - 1)
             (Ops.expand (Ops.unsqueeze order (-2)) (ints (batch @ [ n; n ])))
             v)
          (dtype a))
      (List.tl rotated)
  in
  (Ops.cast w Float64, List.nth_opt v 0)

(* LU with partial pivoting

   Step [j] takes as pivot the first row of largest magnitude in column [j] on
   or below the diagonal, a NaN on the diagonal being the largest and one below
   it the least, and swaps it with row [j], moving each element with its bits.
   The column below the pivot is divided by it, unless it is zero, and the
   trailing rows take the rank-one update. *)

let lu a =
  let dt = dtype a in
  let x = Lower_arith.widen a in
  let batch, m, n = matrix x in
  let rank = Ops.ndim x in
  (* The rows are [int64], as the pivots and the row order are. *)
  let rows = Ops.cast (row_index m) Int64 and cols = column_index n in
  (* [swap u j p] exchanges row [j] and row [p], one per matrix, of [u]. *)
  let swap u j p =
    let _, _, c = matrix u in
    let row_p =
      Lower_index.gather (rank - 2) (Ops.expand p (ints (batch @ [ 1; c ]))) u
    in
    Ops.where (is rows j) row_p (Ops.where (Ops.eq rows p) (row u j) u)
  in
  let step (x, perm, pivots) j =
    let col = column x j in
    let never = float col Float.neg_infinity in
    let magnitude =
      Ops.where
        (Ops.lt rows (Ops.int j))
        never
        (Ops.where (Ops.ne col col)
           (Ops.where (is rows j) (float col Float.infinity) never)
           (Lower_arith.unary Abs col))
    in
    let p = Lower_reduce.arg_reduce Argmax ~axis:(rank - 2) magnitude in
    let pp = Ops.unsqueeze p (-2) in
    let x = swap x j pp and perm = swap perm j pp in
    let pivot = entry x j j and col = column x j in
    let l = Ops.where (Ops.ne pivot (zero pivot)) (fdiv col pivot) col in
    let below = Ops.gt rows (Ops.int j) in
    let x =
      Ops.where
        (Ops.bitwise_and below (is cols j))
        l
        (Ops.where
           (Ops.bitwise_and below (Ops.gt cols (Ops.int j)))
           (Ops.sub x (Ops.mul l (row x j)))
           x)
    in
    (x, perm, p :: pivots)
  in
  let perm = Ops.expand rows (ints (batch @ [ m; 1 ])) in
  let x, perm, pivots =
    List.fold_left step (x, perm, []) (List.init (Int.min m n) Fun.id)
  in
  let perm = Ops.squeeze ~axis:(-1) perm in
  (* The pivots follow no pivot at all, none of [perm]'s elements, which a
     concatenation drops. *)
  let pivots =
    Lower_index.cat (rank - 2)
      (Ops.shrink_to perm (List.map Option.some (ints (batch @ [ 0 ]))))
      (List.rev pivots)
  in
  (Ops.cast x dt, pivots, perm)

(* Cholesky

   Step [j] takes column [j] of the working matrix, on and below the diagonal,
   as column [j] of [L]: its diagonal element's square root heads it, and the
   rest is divided by that root. The working matrix then loses the product of
   the column with itself. Only the lower triangle is ever read. *)

let cholesky ~upper a =
  let dt = dtype a in
  let x = Lower_arith.widen a in
  let _, n, _ = matrix x in
  let rows = row_index n in
  let step (s, columns) j =
    let d = entry s j j in
    let root = Ops.sqrt (Ops.where (Ops.gt d (zero d)) d (float d Float.nan)) in
    let l =
      Ops.where (is rows j)
        (Ops.expand root (Ops.shape (column s j)))
        (Ops.where
           (Ops.gt rows (Ops.int j))
           (fdiv (column s j) root)
           (zero root))
    in
    (Ops.sub s (Ops.mul l (transpose l)), l :: columns)
  in
  let _, columns = List.fold_left step (x, []) (List.init n Fun.id) in
  let l =
    Lower_index.cat
      (Ops.ndim x - 1)
      (block x None (Some (0, 0)))
      (List.rev columns)
  in
  Ops.cast (if upper then transpose l else l) dt

(* Triangular solve

   The system is made lower triangular, [aᵀ] under [transpose], and reversed
   along both axes, with [b]'s rows, when the triangle it reads is the upper
   one. Row [i] of the solution is then row [i] of [b] less the strictly lower
   part of row [i] of the matrix times the rows solved before it, divided by the
   diagonal element. *)

let solve_triangular ~upper ~transpose:t ~unit_diag a b =
  let dt = dtype b in
  let vector = Ops.ndim b = Ops.ndim a - 1 in
  let b = Lower_arith.widen (if vector then Ops.unsqueeze b (-1) else b) in
  let m = Lower_arith.widen (if t then transpose a else a) in
  let rank = Ops.ndim m in
  let reversed = upper <> t in
  let m = if reversed then Ops.flip m [ rank - 2; rank - 1 ] else m in
  let b = if reversed then Ops.flip b [ rank - 2 ] else b in
  let _, n, _ = matrix m in
  let rows = row_index n in
  let strict = Ops.where (Ops.lt (column_index n) rows) m (zero m) in
  let solve x i =
    let rest = Ops.sub (row b i) (dot (row strict i) x) in
    let xi = if unit_diag then rest else fdiv rest (entry m i i) in
    Ops.where (is rows i) (Ops.expand xi (Ops.shape x)) x
  in
  let x = List.fold_left solve (zero b) (List.init n Fun.id) in
  let x = if reversed then Ops.flip x [ rank - 2 ] else x in
  Ops.cast (if vector then Ops.squeeze ~axis:(-1) x else x) dt
