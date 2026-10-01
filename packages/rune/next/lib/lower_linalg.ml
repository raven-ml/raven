(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Tolk_next

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
   below it, applied to [r] on the left and accumulated in [q] on the right. The
   diagonal takes the sign opposite to [x]'s first element, so that no
   cancellation occurs, and a zero column takes no reflection. *)

let sum_last u = Ops.unsqueeze (Ops.rop u Op.Add [ Ops.ndim u - 1 ]) (-1)
let max_last u = Ops.unsqueeze (Ops.rop u Op.Max [ Ops.ndim u - 1 ]) (-1)

let householder a =
  let batch, m, n = matrix a in
  let idx = Ops.arange ~dtype:Int32 m in
  let reflect (q, r) i =
    let at_i = is idx i in
    let c = Ops.squeeze ~axis:(-1) (column r i) in
    let x = Ops.where (Ops.ge idx (Ops.int i)) c (zero c) in
    (* The norm is taken of the column divided by its largest magnitude, so that
       no square underflows or overflows. *)
    let magnitude =
      Ops.where (Ops.lt x (zero x)) (Ops.mul x (float x (-1.))) x
    in
    let largest = max_last magnitude in
    let scale =
      Ops.where (Ops.ne largest (zero largest)) largest (float largest 1.)
    in
    let scaled = fdiv x scale in
    let norm = Ops.mul scale (Ops.sqrt (sum_last (Ops.mul scaled scaled))) in
    let x0 = sum_last (Ops.where at_i x (zero x)) in
    let sgn = direction x0 in
    (* A column already zero below the diagonal takes no reflection, and keeps
       its diagonal element's sign. *)
    let below =
      sum_last (Ops.where (Ops.gt idx (Ops.int i)) magnitude (zero x))
    in
    let active = Ops.ne below (zero below) in
    let u0 = Ops.O.(x0 + (sgn * norm)) in
    let v = fdiv (Ops.where at_i u0 x) (Ops.where active u0 (float u0 1.)) in
    let v = Ops.unsqueeze v (-1) in
    let tau =
      Ops.where active
        (fdiv (Ops.mul sgn u0) (Ops.where active norm (float norm 1.)))
        (zero norm)
    in
    let w = Ops.mul (Ops.unsqueeze tau (-1)) v in
    ( Ops.sub q (dot (dot q v) (transpose w)),
      Ops.sub r (dot w (dot (transpose v) r)) )
  in
  let q = Ops.expand (eye (dtype a) m m) (ints (batch @ [ m; m ])) in
  List.fold_left reflect (q, a) (List.init (Int.min m n) Fun.id)

(* The upper triangle of the matrices of [u], zeros below it. *)
let triu u =
  let _, m, n = matrix u in
  Ops.where (Ops.le (row_index m) (column_index n)) u (zero u)

let qr ~reduced a =
  let dt = dtype a in
  let _, m, n = matrix a in
  let k = Int.min m n in
  let q, r = householder (Lower_arith.widen a) in
  let r = triu r in
  let q, r =
    if reduced then (block q None (Some (0, k)), block r (Some (0, k)) None)
    else (q, r)
  in
  (Ops.cast q dt, Ops.cast r dt)

(* Singular values by one-sided Jacobi rotations

   The columns of [u], [r]'s leading square, are made orthogonal by rotating
   pairs of them, [num / 2] disjoint pairs a round, each by the angle that
   zeroes their inner product; [v] accumulates the rotations. The pairing [p]
   moves as in a round-robin tournament, so that each pair meets once every [num
   - 1] rounds, and a round rotates the pairs [p] selects by one matrix
   product. *)

let pairs num =
  let h = num / 2 in
  Ops.cat
    (Ops.arange ~dtype:Int32 h)
    [ Ops.flip (Ops.arange ~start:h ~dtype:Int32 num) [ 0 ] ]

(* The next round's pairing: every column but the first moves one place. *)
let next_pairs num p =
  if num mod 2 = 1 then Ops.O.((p - int 1) % int num)
  else
    let others = num - 1 in
    let parts = Ops.split p [ 1; others ] in
    Ops.cat (List.nth parts 0)
      [ Ops.O.(((List.nth parts 1 - int 2) % int others) + int 1) ]

(* The rounds that bring [num] columns to orthogonality. A sweep meets every
   pair once, in [num - 1] rounds, or [num] for an odd [num]. The rotations
   converge quadratically once the columns are nearly orthogonal, and the sweeps
   it takes to get there grow as the logarithm of [num]. *)
let rounds num =
  let rec log2 k = if 1 lsl k >= num then k else log2 (k + 1) in
  if num < 2 then 0 else (log2 0 + 3) * (num - 1 + (num mod 2))

let rotate (u, v, p) =
  let batch, num, _ = matrix u in
  let h = num / 2 and dt = dtype u in
  let halves x axis =
    let parts = Ops.split ~axis x [ h; h ] in
    (List.nth parts 0, List.nth parts 1)
  in
  (* The first [h] columns select each pair's first column, the next [h] its
     second. *)
  let selected =
    block
      (Ops.cast (Ops.eq (row_index num) (Ops.unsqueeze p 0)) dt)
      None
      (Some (0, 2 * h))
  in
  let paired = dot u selected in
  let left, right = halves paired (-1) in
  let column_sums x = Ops.rop x Op.Add [ Ops.ndim x - 2 ] in
  let gamma =
    Ops.reshape (column_sums Ops.O.(left * right)) (ints (batch @ [ 1; h ]))
  in
  let alpha, beta =
    halves (Ops.unsqueeze (column_sums Ops.O.(paired * paired)) (-2)) (-1)
  in
  let rot = Ops.ne gamma (zero gamma) in
  let one = float gamma 1. and two = float gamma 2. in
  let tau = fdiv Ops.O.(beta - alpha) Ops.O.(two * Ops.where rot gamma one) in
  let t =
    fdiv (direction tau)
      Ops.O.(Lower_arith.unary Abs tau + Ops.sqrt (one + (tau * tau)))
  in
  let t = Ops.where rot t (zero t) in
  let c = fdiv one (Ops.sqrt Ops.O.(one + (t * t))) in
  let s = Ops.O.(c * t) in
  let mi, mj = halves (transpose selected) (-2) in
  let col x = Ops.unsqueeze x (-1) and row x = Ops.unsqueeze x (-2) in
  let per_pair x = Ops.reshape x (ints (batch @ [ h; 1; 1 ])) in
  let cc = per_pair (Ops.sub c (float c 1.)) and ss = per_pair s in
  let delta =
    Ops.O.(
      (cc * ((col mi * row mi) + (col mj * row mj)))
      + (ss * ((col mi * row mj) - (col mj * row mi))))
  in
  let r =
    Ops.add (eye dt num num) (Ops.rop delta Op.Add [ Ops.ndim delta - 3 ])
  in
  (dot u r, dot v r, next_pairs num p)

let svd ~full_matrices a =
  let dt = dtype a in
  let batch, m, n = matrix a in
  let num = Int.min m n and q_num = Int.max m n in
  let x = Lower_arith.widen a in
  let wdt = dtype x in
  let q, r = householder (if m >= n then x else transpose x) in
  let square k = Ops.expand (eye wdt k k) (ints (batch @ [ k; k ])) in
  let u = Ops.contiguous (block (triu r) (Some (0, num)) (Some (0, num))) in
  let v = Ops.contiguous (square num) in
  let u, v, _ =
    List.fold_left
      (fun uvp _ -> rotate uvp)
      (u, v, pairs num)
      (List.init (rounds num) Fun.id)
  in
  let r = Ops.ndim u in
  let norms = Ops.sqrt (Ops.rop (Ops.mul u u) Op.Add [ r - 2 ]) in
  let order = Lower_reduce.argsort ~descending:true ~axis:(r - 2) norms in
  let s = Lower_index.gather (r - 2) order norms in
  let by_order x =
    Lower_index.gather (r - 1)
      (Ops.expand (Ops.unsqueeze order (-2)) (ints (batch @ [ num; num ])))
      x
  in
  (* The sorted columns are orthogonal: the reflections that triangularize them
     are an orthogonal basis of which each column, with the sign it takes on the
     diagonal, is the direction, and which completes the columns of zero
     norm. *)
  let q_u, r_u = householder (by_order u) in
  let diagonal =
    Ops.rop
      (Ops.where (Ops.eq (row_index num) (column_index num)) r_u (zero r_u))
      Op.Add
      [ r - 2 ]
  in
  let u = Ops.mul q_u (Ops.unsqueeze (direction diagonal) (-2)) in
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
  let rows = row_index m and cols = column_index n in
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
